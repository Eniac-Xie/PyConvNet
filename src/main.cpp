#include <iostream>
#include <string>
#include<fstream>

using namespace std;

int main()

{
    ifstream file ( "../test/sample_data/X.csv" );
    string value;

    int i = 0;
    while ( file.good() && i < 100)
    {
        getline ( file, value, ',' );
        cout << string( value, 0, value.length() )<<endl;
        ++i;
    }
}