# PhantomX Integration Guide

## Integrating LaBraM-POYO with PhantomLink Streaming

This guide shows how to use the LaBraM decoder in PhantomLink's real-time streaming pipeline.

### Quick Integration

#### 1. Add LaBraM Middleware to PlaybackEngine

Edit `phantomlink/playback_engine.py`:

```python
from phantomlink.labram_integration import LabramDecoderMiddleware

class PlaybackEngine:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # Add LaBraM decoder (optional)
        try:
            self.labram_decoder = LabramDecoderMiddleware(
                model_path="../PhantomX/models/best_model.pt",
                use_tta=True  # Enable test-time adaptation
            )
            logger.info("LaBraM decoder enabled")
        except Exception as e:
            logger.warning(f"LaBraM decoder not available: {e}")
            self.labram_decoder = None
```

#### 2. Decode Packets in Stream

In the `stream()` method:

```python
async def stream(self) -> AsyncGenerator[StreamPacket, None]:
    """Stream packets with LaBraM predictions."""
    
    for i in range(len(self.loader.spike_counts)):
        # ... existing packet creation ...
        
        # Add LaBraM prediction (if available)
        if self.labram_decoder is not None:
            spike_counts = np.array(packet.spikes.spike_counts)
            labram_pred = self.labram_decoder.decode_packet(spike_counts)
            
            # Add to packet metadata or as separate field
            packet.labram_velocity = labram_pred
        
        yield packet
```

### Advanced Usage

#### Test-Time Adaptation with Drift Simulation

Combine with `NoiseInjectionMiddleware` to test TTA:

```python
# In playback_engine.py
noise_middleware = NoiseInjectionMiddleware(
    drift_amplitude=0.3,
    drift_period_seconds=60.0
)

labram_decoder = LabramDecoderMiddleware(
    model_path="models/best_model.pt",
    use_tta=True,
    tta_strategy='entropy'
)

# Apply noise, then decode with TTA
noisy_packet = noise_middleware.inject_noise(packet, elapsed_time)
labram_pred = labram_decoder.decode_packet(noisy_packet.spikes.spike_counts)
```

#### Visualization in PhantomLoop

Send LaBraM predictions to PhantomLoop WebSocket:

```python
# In websocket handler
packet_dict = {
    'spikes': packet.spikes.dict(),
    'kinematics': packet.kinematics.dict(),
    'labram_prediction': packet.labram_velocity,  # New field
    'timestamp': packet.metadata.timestamp_sec
}

await websocket.send_json(packet_dict)
```

Update PhantomLoop to display LaBraM predictions alongside ground truth.

### Performance Considerations

#### Latency Budget

- **Target**: <10ms per packet (to maintain 40Hz streaming)
- **LaBraM decoder**: ~1-3ms on CPU, <1ms on GPU
- **Tokenization**: <0.5ms
- **TTA overhead**: +1-2ms per packet

#### Optimization Tips

1. **Batch Processing**: Process multiple packets together
   ```python
   # Collect 10 packets
   batch_spikes = np.array([p.spikes.spike_counts for p in packet_buffer])
   batch_predictions = labram_decoder.decoder.decode_batch(batch_spikes)
   ```

2. **GPU Acceleration**: Use CUDA if available
   ```python
   decoder = LabramDecoderMiddleware(model_path, device='cuda')
   ```

3. **Disable TTA for latency-critical paths**
   ```python
   decoder = LabramDecoderMiddleware(model_path, use_tta=False)
   ```

### Testing Integration

#### Unit Test

```python
# tests/test_labram_integration.py
from phantomlink.labram_integration import LabramDecoderMiddleware
import numpy as np

def test_decoder_latency():
    decoder = LabramDecoderMiddleware("models/test_model.pt")
    
    # Test 1000 packets
    for _ in range(1000):
        spikes = np.random.poisson(2.0, size=142)
        velocity = decoder.decode_packet(spikes)
    
    stats = decoder.get_statistics()
    assert stats['mean_latency_ms'] < 10.0, "Latency too high!"
```

#### Integration Test with Streaming

```bash
# Start PhantomLink with LaBraM decoder
python -m phantomlink.main --enable-labram --model-path models/best_model.pt

# Should see in logs:
# INFO: LaBraM decoder loaded successfully
# INFO: Codebook size: 256
# INFO: Streaming at 40Hz with LaBraM predictions
```

### Troubleshooting

#### Import Errors

If `ImportError: No module named 'phantomx'`:

```bash
cd ../PhantomX/python
pip install -e .
```

#### High Latency

Check decoder statistics:

```python
stats = middleware.get_statistics()
print(f"Mean latency: {stats['mean_latency_ms']:.2f} ms")
print(f"Packets processed: {stats['middleware_packet_count']}")
```

If >10ms, consider:
- Moving to GPU (`device='cuda'`)
- Disabling TTA
- Using smaller codebook (128 instead of 256)

#### TTA Instability

If predictions become unstable with TTA:

```python
# Reset TTA state periodically
if elapsed_time % 30.0 < 0.025:  # Every 30 seconds
    middleware.reset_tta()
```

### API Reference

See [`labram_integration.py`](../PhantomLink/src/phantomlink/labram_integration.py) for full API documentation.

---

**Next Steps:**
1. Train your first codebook: `python train_labram.py --data_path data/mc_maze.nwb`
2. Test integration: `python -m phantomlink.labram_integration --model_path models/best_model.pt`
3. Enable in production: Add middleware to `playback_engine.py`
