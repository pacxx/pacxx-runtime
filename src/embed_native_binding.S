	.global native_binding_start
native_binding_start:
        .incbin FILE
        .global native_binding_end
native_binding_end:
	.byte 0

        .global native_binding_size
native_binding_size:
        .int native_binding_end - native_binding_start
