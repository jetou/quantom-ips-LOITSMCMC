from quantom_ips import register

# Base:
register(
    id="Base",
    entry_point="quantom_ips.gradient_transport.torch_base_gradient_transport:TorchBaseGradientTransport",
    group="gradient_transport",
)
# ARAR
register(
    id="ARAR",
    entry_point="quantom_ips.gradient_transport.torch_arar:TorchARAR",
    group="gradient_transport",
)
# ARAR with chunks:
register(
    id="ChunkARAR",
    entry_point="quantom_ips.gradient_transport.torch_chunk_arar:TorchChunkARAR",
    group="gradient_transport",
)
# Dual Binary Tree:
register(
    id="DualBinaryTree",
    entry_point="quantom_ips.gradient_transport.torch_dual_binary_tree:TorchDualBinaryTree",
    group="gradient_transport",
)
