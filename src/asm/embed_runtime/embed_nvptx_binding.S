	.global nvptx_binding_start
nvptx_binding_start:
        .incbin FILE
        .global nvptx_binding_end
nvptx_binding_end:
	.byte 0

        .global nvptx_binding_size
nvptx_binding_size:
        .int nvptx_binding_end - nvptx_binding_start
