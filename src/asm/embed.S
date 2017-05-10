#define CONCAT(x, y)   x ## y
#define TAG_START(tag) CONCAT(tag, _start)
#define TAG_END(tag)   CONCAT(tag, _end)

	.data
	.global TAG_START(TAG)
TAG_START(TAG):
        .incbin FILE
    .global TAG_END(TAG)
TAG_END(TAG):
	.byte 0