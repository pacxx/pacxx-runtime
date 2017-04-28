	.global llvm_start
llvm_start:
        .incbin FILE
        .global llvm_end
llvm_end:    
	.byte 0

        .global llvm_size
llvm_size:
        .int llvm_end - llvm_start
