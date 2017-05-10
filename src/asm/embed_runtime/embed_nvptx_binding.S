	.data
	.global nvptx_binding_start
nvptx_binding_start:
        .incbin FILE
        .global nvptx_binding_end
nvptx_binding_end:
	.byte 0
