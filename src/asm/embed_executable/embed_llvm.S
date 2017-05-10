	.data
	.global llvm_start
llvm_start:
        .incbin FILE
    .global llvm_end
llvm_end:    
	.byte 0