import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_size, hid_size, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hid_size*2, hid_size)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        
        h_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        c_cat = torch.cat([cell[-2], cell[-1]], dim=1)
        h_final = torch.tanh(self.fc(h_cat)).unsqueeze(0)
        c_final = torch.tanh(self.fc(c_cat)).unsqueeze(0)

        return outputs, (h_final, c_final)


class Attention(nn.Module):
    def __init__(self, hid_size):
        super().__init__()
        self.attn = nn.Linear(hid_size*3, hid_size)
        self.v = nn.Linear(hid_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, enc_hid_size, dec_hid_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_ID)
        self.attention = Attention(dec_hid_size)
        self.lstm = nn.LSTM(emb_size + enc_hid_size, dec_hid_size,
                            n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(
            dec_hid_size + enc_hid_size + emb_size, vocab_size)

    def forward_step(self, input_tok, hidden, cell, encoder_outputs, mask):
        embedded = self.embedding(input_tok).unsqueeze(1)
        h = hidden[-1]
        attn_weights = self.attention(h, encoder_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        rnn_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        logits = self.out(torch.cat([output, context, embedded], dim=1))
        return logits, hidden, cell

    def forward(self, tgt, hidden, cell, encoder_outputs, mask, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(0)
        max_len = tgt.size(1)
        vocab_size = self.out.out_features

        outputs = torch.zeros(batch_size, max_len,
                              vocab_size, device=tgt.device)
        input_tok = tgt[:, 0]  # <sos>
        for t in range(1, max_len):
            logits, hidden, cell = self.forward_step(
                input_tok, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = logits
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def create_mask(self, src):
        return (src != PAD_ID)

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        encoder_outputs, (h, c) = self.encoder(src, src_lengths)
        mask = self.create_mask(src)
        outputs = self.decoder(tgt, h, c, encoder_outputs,
                               mask, teacher_forcing_ratio)
        return outputs
