#include <typeinfo>

#include "lil_name_gen.hpp"
#include "lil_name_nn.hpp"

int main() {
    // NameGenerator lil_name_gen("names.txt");
    // lil_name_gen.generate(20, 2147483647);

    LilNameNN lil_name_nn(2147483647);
    lil_name_nn.train(100);
    lil_name_nn.sample(10);



    return 0;
}