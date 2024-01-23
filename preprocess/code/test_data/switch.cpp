#include <iostream>
using namespace std;

int num = 0;

int main() {
    switch (num) {
    case 0:
        cout << "0" << endl;
    case 1:
        cout << "1" << endl;
    case 2:
        cout << "2" << endl;
    default:
        cout << "default" << endl;
    }
}
