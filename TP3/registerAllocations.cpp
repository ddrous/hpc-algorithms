#include <iostream>
#include <fstream>
#include <string>
using namespace std;

/* I am a comment */

/* I am a multine
comment */

/** I am a difficult
multiline
comment *
with star
*/
int main () {
  string line;
  ifstream myfile ("example.txt");
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      cout << line << '\n';
    }
    myfile.close();
  }
  else cout << "Unable to open file"; 

  double num = 1.2;
  
  float anotherNum = 1123123.3123123E-1;

  return 0;
}