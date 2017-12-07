#pragma once
#include <vector>
namespace pacxx {
	namespace v2 {
		class Executor;
		std::vector<Executor> * getExecutorMemory();
		void registerModule(const char *start, const char *end);
	}
}