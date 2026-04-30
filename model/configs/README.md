# Supported RibCXR Configs

This directory contains only the RibCXR YAML configs intentionally supported by the web inference app.

Do not copy every training config from `/home/zijianguo/Code/MIDL2021-VinDr-RibCXR/cvcore/config` into this repository. Add a config here only when there is a matching checkpoint and the app should support that model variant.

Current supported config:

- `multi_unetpp_b0_dice.yaml`: U-Net++ with EfficientNet-B0 encoder, 1-channel CXR input, 20 rib output channels.

Configs in this directory should contain only inference-required fields. Training-only sections such as dataset JSON paths, loss, metrics, optimizer, scheduler, batch size, epochs, workers, and random seed are intentionally omitted.
