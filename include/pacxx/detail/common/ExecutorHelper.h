#pragma once
#include <vector>
#include <cstddef>

namespace llvm {
	class Module;
}

namespace pacxx {
	namespace v2 {
		class Executor;
		std::vector<Executor> * getExecutorMemory();
		void registerModule(const char *start, const char *end);

		void initializeModule(Executor& exec);
		void initializeModule(Executor &exec, const char* ptr, std::size_t size);
	}
}