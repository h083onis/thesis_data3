#include <iostream>

int main() {
    int i = 0;

    // do-while文でdoの後に中括弧なし
    do
        std::cout << i++ << " ";
    while (i < 5);

    // 改行
    std::cout << std::endl;

    return 0;
}
