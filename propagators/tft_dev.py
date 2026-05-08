import torch
from .tft import PropagatorTFT as BasePropagatorTFT

class PropagatorTFT(BasePropagatorTFT):
    
    def forward(self, data):
        with torch.no_grad():
            encdec_out, latent = self.encdec_model(data)
        
        latent = self.normalize(latent)

        long_seq_length = self.trainer.datamodule.sequence_length # This is the length of one input sequence
        prop_latent = latent.view(-1, long_seq_length, self.latent_dim)
        prop_in = prop_latent[:, :self.propagator.input_chunk_length, :]
        n_chunks = (long_seq_length - self.propagator.input_chunk_length) // self.propagator.output_chunk_length
        
        prop_out = None
        for _ in range(n_chunks):
            # TFT expects (B, T, C) inputs; returns (B, T_out, C)
            prop_out_chunk = self.propagator((prop_in, None, None)) # (Batch, T_out, Latent Dim)
            if prop_out is None:
                prop_out = prop_out_chunk
            else:
                prop_out = torch.cat([prop_out, prop_out_chunk], dim=1)

            if prop_out.size(1) >= self.propagator.input_chunk_length:
                prop_in = self.prop_sample(prop_out)[:, -self.propagator.input_chunk_length:, :]
            else:
                prop_in = torch.cat([prop_in, self.prop_sample(prop_out)], dim=1)[:, -self.propagator.input_chunk_length:, :]
        self.propagator.reset_cell_state()
        prop_samp = self.prop_sample(prop_out.clone())
        prop_samp = prop_samp.view(-1, self.latent_dim)
        prop_samp = self.denormalize(prop_samp)
        prop_dec = self.encdec_model.decode(prop_samp)
        return encdec_out, latent, prop_out, prop_dec
    