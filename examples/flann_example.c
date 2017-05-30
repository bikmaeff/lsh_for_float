

#include <flann/flann.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float* read_points(const char* filename, int rows, int cols)
{
	float* data;
	float *p;
	FILE* fin;
	int i,j;

    fin = fopen(filename,"r");
    if (!fin) {
        printf("Cannot open input file.\n");
        exit(1);
    }
    
    data = (float*) malloc(rows*cols*sizeof(float));
    if (!data) {
        printf("Cannot allocate memory.\n");
        exit(1);
    }
    p = data;
    
    for (i=0;i<rows;++i) {
        for (j=0;j<cols;++j) {
            fscanf(fin,"%g ",p);
            p++;
        }
    }
    
    fclose(fin);
    
    return data;
}

void write_results(char* filename, int *data, int rows, int cols)
{
	FILE* fout;
	int* p;
	int i,j;

    fout = fopen(filename,"w");
    if (!fout) {
        printf("Cannot open output file.\n");
        exit(1);
    }
    
    p = data;
    for (i=0;i<rows;++i) {
        for (j=0;j<cols;++j) {
            fprintf(fout,"%d ",*p);
            p++;
        }
        fprintf(fout,"\n");
    }
    fclose(fout);
}
void write_results_float(char* filename, float *data, int rows, int cols)
{
	FILE* fout;
	float* p;
	int i,j;

    fout = fopen(filename,"w");
    if (!fout) {
        printf("Cannot open output file.\n");
        exit(1);
    }
    
    p = data;
    for (i=0;i<rows;++i) {
        for (j=0;j<cols;++j) {
            fprintf(fout,"%f ",*p);
            p++;
        }
        fprintf(fout,"\n");
    }
    fclose(fout);
}



int main(int argc, char** argv)
{
	float* dataset;
	float* testset;
	int nn;//how many approxi nearest neighbor need to find for each query point
	int* result;
	float* dists;
	struct FLANNParameters p;
	float speedup;
	flann_index_t index_id;
	char resultname[100] = "C_default.error";
	char distname[100] = "C_defaultdist.error";
	
    int rows = 9000;
    int cols = 128;
    int tcount = 1000;

    /*
     * The files dataset.dat and testset.dat can be downloaded from:
     * http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/dataset.dat
     * http://people.cs.ubc.ca/~mariusm/uploads/FLANN/datasets/testset.dat
     */
    printf("Reading input data file.\n");
    dataset = read_points("dataset.dat", rows, cols);
    printf("Reading test data file.\n");
    testset = read_points("testset.dat", tcount, cols);
    
    nn = 3;
    result = (int*) malloc(tcount*nn*sizeof(int));
    dists = (float*) malloc(tcount*nn*sizeof(float));
    
    p = DEFAULT_FLANN_PARAMETERS;
    p.algorithm = FLANN_INDEX_KDTREE;
    p.trees = 8;
    p.log_level = FLANN_LOG_INFO;
	p.checks = 64;
    
    printf("Computing index.\n");
    index_id = flann_build_index(dataset, rows, cols, &speedup, &p);
    printf("how many result point for each query?\n");
    scanf("%d", &nn);
    sprintf(resultname, "results_%dnn.dat", nn);
    sprintf(distname, "dists_%dnn.dat", nn);
    fflush(stdout);
    flann_find_nearest_neighbors_index(index_id, testset, tcount, result, dists, nn, &p);
    write_results(resultname, result, tcount, nn); 
    write_results_float(distname, dists, tcount, nn); 
    printf("result file name: %s", resultname);
    
    flann_free_index(index_id, &p);
    free(dataset);
    free(testset);
    free(result);
    free(dists);
    
    return 0;
}
