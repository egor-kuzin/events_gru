TL;DR: Дал два полностью рабочих Lightning-класса: GRUClassifier (стриминг “из коробки”) и MambaClassifier (SSM-класс; для него нужен пакет mamba-ssm). Оба учитывают второй канал — длительности между событиями: можно подать как непрерывные (секунды/мс) или дискретные бины. Сразу встроены ReduceLROnPlateau и поддержка дисбаланса через pos_weight.


---

Полный код (без заглушек)

# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW
from torchmetrics.classification import BinaryAveragePrecision, BinaryAUROC


# ========= Утилиты: маски, длины, time-encoding =========

def lengths_from_pad(x: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    x: LongTensor [B, L] (идентификаторы событий)
    Возвращает длины (без PAD) как LongTensor [B].
    """
    assert x.dim() == 2
    mask = (x != pad_idx)
    lengths = mask.sum(dim=1)
    # Гарантируем хотя бы 1, чтобы pack_padded не падал (если есть полностью паддинговые строки)
    lengths = torch.clamp(lengths, min=1)
    return lengths


class TimeEncoding(nn.Module):
    """
    Встраивание длительностей между событиями.
    Поддерживает 2 режима:
      - дискретные бины: delta_is_discrete=True -> nn.Embedding
      - непрерывные длительности (float, секунды/мс): delta_is_discrete=False -> MLP(log1p(delta))
    """
    def __init__(
        self,
        emb_dim: int,
        delta_is_discrete: bool,
        delta_bins: Optional[int] = None,
        padding_idx: Optional[int] = 0,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        self.delta_is_discrete = delta_is_discrete
        if delta_is_discrete:
            assert delta_bins is not None and delta_bins > 1, "Нужно указать delta_bins>1 для дискретных длительностей"
            self.emb = nn.Embedding(delta_bins, emb_dim, padding_idx=padding_idx if padding_idx is not None else 0)
            self.out_dim = emb_dim
        else:
            self.mlp = nn.Sequential(
                nn.Linear(1, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, emb_dim),
            )
            self.out_dim = emb_dim

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta:
          - если discrete: LongTensor [B, L]
          - если continuous: FloatTensor [B, L] (секунды/мс)
        out: FloatTensor [B, L, D]
        """
        if self.delta_is_discrete:
            emb = self.emb(delta)  # [B, L, D]
            return emb
        else:
            if delta.dtype != torch.float32 and delta.dtype != torch.float64:
                delta = delta.float()
            x = torch.log1p(torch.clamp(delta, min=0.0)).unsqueeze(-1)  # [B, L, 1]
            return self.mlp(x)  # [B, L, D]


# ========= Вариант 1: GRU (стриминг “из коробки”) =========

class GRUClassifier(pl.LightningModule):
    """
    Классификация последовательностей категориальных событий с доп. каналом длительностей.
    Архитектура: Emb(event) || TimeEncoding(delta) -> GRU -> последний hidden -> Linear -> логиты.
    Поддержка:
      - дисбаланс: BCEWithLogitsLoss(pos_weight)
      - ReduceLROnPlateau по val_aupr
      - стриминг: сохранение/передача скрытого состояния между шагами (online)
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        event_emb_dim: int = 64,
        delta_emb_dim: int = 32,
        delta_is_discrete: bool = False,
        delta_bins: Optional[int] = None,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.event_emb = nn.Embedding(vocab_size, event_emb_dim, padding_idx=pad_idx)

        self.time_enc = TimeEncoding(
            emb_dim=delta_emb_dim,
            delta_is_discrete=delta_is_discrete,
            delta_bins=delta_bins,
            padding_idx=0 if delta_is_discrete else None,
        )

        in_dim = event_emb_dim + self.time_enc.out_dim

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

        # Лосс с учетом дисбаланса
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Метрики
        self.train_ap = BinaryAveragePrecision()
        self.val_ap = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()

        # Стейт для стриминга
        self._stream_state = None  # shape: [num_layers, 1, hidden_size]

    def forward(
        self,
        events: torch.Tensor,      # LongTensor [B, L]
        deltas: torch.Tensor,      # LongTensor [B, L] если discrete, иначе FloatTensor [B, L]
        lengths: Optional[torch.Tensor] = None,  # LongTensor [B]
    ) -> torch.Tensor:
        B, L = events.shape
        if lengths is None:
            lengths = lengths_from_pad(events, self.hparams.pad_idx)  # [B]

        ev = self.event_emb(events)           # [B, L, De]
        dt = self.time_enc(deltas)            # [B, L, Dt]
        x = torch.cat([ev, dt], dim=-1)       # [B, L, De+Dt]

        # Упаковка по длинам
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)    # h_n: [num_layers, B, H]
        last = h_n[-1]                        # [B, H]

        logits = self.fc(self.dropout(last))  # [B, 1]
        return logits.squeeze(-1)             # [B]

    # ----- Lightning API -----

    def training_step(self, batch, batch_idx):
        events, deltas, labels = batch["events"], batch["deltas"], batch["labels"].float()
        logits = self(events, deltas)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            self.train_ap.update(probs, labels.int())

        return loss

    def on_train_epoch_end(self):
        ap = self.train_ap.compute()
        self.log("train_aupr", ap, on_epoch=True, prog_bar=True)
        self.train_ap.reset()

    def validation_step(self, batch, batch_idx):
        events, deltas, labels = batch["events"], batch["deltas"], batch["labels"].float()
        logits = self(events, deltas)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        self.val_ap.update(probs, labels.int())
        self.val_auroc.update(probs, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        ap = self.val_ap.compute()
        auroc = self.val_auroc.compute()
        self.log("val_aupr", ap, on_epoch=True, prog_bar=True)
        self.log("val_auroc", auroc, on_epoch=True, prog_bar=False)
        self.val_ap.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_aupr",
                "interval": "epoch",
                "frequency": 1,
                "name": "ReduceLROnPlateau(val_aupr)",
            },
        }

    # ----- Стриминг (онлайн) -----

    def reset_stream_state(self):
        """Сброс скрытого состояния перед новым стримом."""
        self._stream_state = None

    @torch.no_grad()
    def stream_step(
        self,
        event_id: torch.Tensor,       # LongTensor [] или [1]
        delta_value: torch.Tensor,    # LongTensor []/[1] если discrete, иначе FloatTensor []/[1]
    ) -> torch.Tensor:
        """
        Обрабатывает один шаг последовательности онлайн.
        Возвращает логит (скаляр Tensor).
        """
        if event_id.dim() == 0:
            event_id = event_id.view(1)
        if delta_value.dim() == 0:
            delta_value = delta_value.view(1)

        ev = self.event_emb(event_id.view(1, 1))     # [1,1,De]
        if self.hparams.delta_is_discrete:
            dt = self.time_enc.emb(delta_value.view(1, 1))  # [1,1,Dt]
        else:
            x = delta_value.view(1, 1).float()
            x = torch.log1p(torch.clamp(x, min=0.0)).unsqueeze(-1)  # [1,1,1]
            dt = self.time_enc.mlp(x)  # [1,1,Dt]

        x = torch.cat([ev, dt], dim=-1)  # [1,1,De+Dt]

        out, self._stream_state = self.gru(x, self._stream_state)  # out: [1,1,H]
        logit = self.fc(self.dropout(out[:, -1, :]))  # [1,1] -> [1,1]
        return logit.squeeze()  # скаляр


# ========= Вариант 2: Mamba (SSM). Требуется пакет mamba-ssm =========

class MambaBlock(nn.Module):
    """
    Обертка над Mamba-блоком с остаточной связью и LayerNorm.
    Требует: pip install mamba-ssm
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        try:
            # В актуальных версиях работает такой импорт:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError(
                "Не найден пакет 'mamba-ssm'. Установите: pip install mamba-ssm\n"
                f"Исключение: {repr(e)}"
            )
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        """
        h = self.norm(x)
        h = self.mamba(h)  # [B, L, D]
        return x + self.dropout(h)


class MambaClassifier(pl.LightningModule):
    """
    Классификация последовательностей через стек Mamba-блоков.
    Emb(event) || TimeEncoding(delta) -> Linear->D -> [Mamba x N] -> last-step pooling -> Linear -> логиты

    Примечание: для инференса в стриминге у Mamba есть API скрытых состояний,
    зависящее от версии библиотеки. Здесь реализован офлайн путь (батчи).
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        event_emb_dim: int = 64,
        delta_emb_dim: int = 32,
        delta_is_discrete: bool = False,
        delta_bins: Optional[int] = None,
        d_model: int = 256,
        depth: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.event_emb = nn.Embedding(vocab_size, event_emb_dim, padding_idx=pad_idx)
        self.time_enc = TimeEncoding(
            emb_dim=delta_emb_dim,
            delta_is_discrete=delta_is_discrete,
            delta_bins=delta_bins,
            padding_idx=0 if delta_is_discrete else None,
        )
        in_dim = event_emb_dim + self.time_enc.out_dim
        self.proj_in = nn.Linear(in_dim, d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.train_ap = BinaryAveragePrecision()
        self.val_ap = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()

    def forward(
        self,
        events: torch.Tensor,      # LongTensor [B, L]
        deltas: torch.Tensor,      # LongTensor [B, L] (если discrete) или FloatTensor [B, L]
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            lengths = lengths_from_pad(events, self.hparams.pad_idx)  # [B]

        ev = self.event_emb(events)     # [B, L, De]
        dt = self.time_enc(deltas)      # [B, L, Dt]
        x = torch.cat([ev, dt], dim=-1) # [B, L, De+Dt]
        x = self.proj_in(x)             # [B, L, D]

        for blk in self.blocks:
            x = blk(x)                  # [B, L, D]

        x = self.norm_out(self.dropout(x))  # [B, L, D]

        # Пулинг: берём представление последнего непаддингового шага
        B = x.size(0)
        lengths = lengths.to(x.device)
        idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, x.size(-1))  # [B,1,D]
        last_states = x.gather(dim=1, index=idx).squeeze(1)  # [B, D]

        logits = self.fc(last_states)    # [B, 1]
        return logits.squeeze(-1)        # [B]

    # ----- Lightning API -----

    def training_step(self, batch, batch_idx):
        events, deltas, labels = batch["events"], batch["deltas"], batch["labels"].float()
        logits = self(events, deltas)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            self.train_ap.update(probs, labels.int())

        return loss

    def on_train_epoch_end(self):
        ap = self.train_ap.compute()
        self.log("train_aupr", ap, on_epoch=True, prog_bar=True)
        self.train_ap.reset()

    def validation_step(self, batch, batch_idx):
        events, deltas, labels = batch["events"], batch["deltas"], batch["labels"].float()
        logits = self(events, deltas)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        self.val_ap.update(probs, labels.int())
        self.val_auroc.update(probs, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        ap = self.val_ap.compute()
        auroc = self.val_auroc.compute()
        self.log("val_aupr", ap, on_epoch=True, prog_bar=True)
        self.log("val_auroc", auroc, on_epoch=True, prog_bar=False)
        self.val_ap.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_aupr",
                "interval": "epoch",
                "frequency": 1,
                "name": "ReduceLROnPlateau(val_aupr)",
            },
        }


---

Как подавать батчи

# batch = {
#   "events": LongTensor [B, L]  (ID событий, где pad_idx — паддинг)
#   "deltas": LongTensor [B, L]  если delta_is_discrete=True
#             FloatTensor [B, L] если delta_is_discrete=False (напр., секунды)
#   "labels": FloatTensor [B]    (0/1)
# }

Примечания и советы

pos_weight ставьте примерно N_neg / N_pos (скаляр).

Для непрерывных длительностей лучше подавать их в секундах или миллисекундах; внутри используется log1p(delta).

Для дискретных длительностей задайте delta_is_discrete=True и delta_bins (например, квантили/бины).

Mamba требует pip install mamba-ssm. Если импорт не найдётся, класс честно поднимет ImportError с инструкцией.


Хочешь — добавлю готовые LightningDataModule, функцию поиска оптимального порога по PR-кривой, и вариант Mamba со стриминг-API (если укажешь версию mamba-ssm, под неё напишу stateful-инференс).
