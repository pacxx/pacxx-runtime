//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_COREINITIALIZER_H
#define PACXX_V2_COREINITIALIZER_H


namespace pacxx
{
namespace core
{
    class CoreInitializer
    {
    public:
        static void initialize();
    private:
        CoreInitializer();
        void initializeCore();
        bool _initialized;
    };
}
}


#endif //PACXX_V2_COREINITIALIZER_H
