//
// Created by lars on 18/11/16.
//

#ifndef PACXX_V2_SPMDVECTORIZER_H
#define PACXX_V2_SPMDVECTORIZER_H

namespace pacxx {
    namespace v2 {

        class SPMDVectorizer {

        public:
            bool vectorize(llvm::Module* module);


        private:

            llvm::TargetMachine* _machine;
            llvm::legacy::FunctionPassManager _FPM;
        };
    }
}


#endif //PACXX_V2_SPMDVECTORIZER_H
