#include <iostream>
#include <file1.h>

using namespace std;

int main() {
  extern int i;
  cout << "Hello World!"<< endl;
  cout<<i<<endl;
  return 0;
}