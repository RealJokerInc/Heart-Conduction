// =========================================================================================================
// Author: Lucas Berg
// =========================================================================================================
// Program that reads a Purkinje network in VTP format convert it to a graph data structure and extract its
// terminals.
// =========================================================================================================

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>

#include "purkinje/purkinje.h"
#include "graph/graph.h"

#define PRINT_LINE "============================================================================="

using namespace std;

int main (int argc, char *argv[])
{
    if (argc-1 != 1)
    {
        cerr << PRINT_LINE << endl;
        cerr << "Usage:> " << argv[0] << " <vtp_file>" << endl;
        cerr << PRINT_LINE << endl;
        cerr << "<vtp_file> = VTP file with the solution of a timestep from the simulation" << endl;
        cerr << PRINT_LINE << endl;
        cerr << "Example:" << endl;
        cerr << argv[0] << " inputs/activation-map.vtp" << endl;
        cerr << PRINT_LINE << endl;
        
        exit(EXIT_FAILURE);
    }

    string filename = argv[1];

    // Read all the data from the file
    vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    vtkPolyData* polydata = reader->GetOutput();

    struct purkinje_network *the_purkinje_network = new_purkinje_network();
    read_purkinje_network_from_vtp(the_purkinje_network,polydata);
    //print_purkinje_network(the_purkinje_network);

    struct graph *the_graph = new_graph();
    build_graph_from_purkinje_network(the_graph,the_purkinje_network);

    uint32_t num_terminals;
    struct terminal *the_terminals;
    the_terminals = calculate_terminals(the_graph,num_terminals);
    //print_terminals(the_terminals,num_terminals);

    return 0;
}
