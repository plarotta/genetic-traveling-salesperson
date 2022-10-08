//C++ implementation of the GA for the Travelign Salesman Problem


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;



// this will read in the rad data and create the list of cities
pair<double,double>* read_data(){ 
    vector<string> my_data; 
    ifstream myfile; 
    myfile.open("tsp.txt");
    string myline;
    while(getline(myfile, myline)){
        //std::cout << myline;
        my_data.push_back(myline);
    }
    myfile.close();
        
    double res1;
    double res2;
    string string1; 
    string string2;
    static pair<double,double> pairs[1000];

    for (int i=0; i<my_data.size(); ++i){
        string1 = my_data[i].substr(0,6);
        const char *charred1 = string1.c_str();
        res1 = atof(charred1);
        string2 = my_data[i].substr(9,6);
        const char *charred2 = string2.c_str();
        res2 = atof(charred2);
        pairs[i] = make_pair(res1,res2);
    }  
    return(pairs);
} 

void get_fitness(){} // this will get my path lengths

void mutate(){} //this will insert a mutation, meaning swap two cities

void cross_parents(){} //this will implement crossover

void mutation_chance(){} //this will decide if a mutation will happen based on some specified mutation rate

void crossover_chance(){} //this will decide if a crossover (mating) will happen between two parents based on some specified crossover rate

void random_search(){}

int main(){

    pair<double,double> city_locations [1000];
    city_locations = read_data();
    cout << city_locations[0].first;

    return 0;
}