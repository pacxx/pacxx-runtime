	.global reflection_start
reflection_start:
        .incbin FILE
        .global reflection_end
reflection_end:    
	.byte 0

        .global reflection_size
reflection_size:
        .int reflection_end - reflection_start
