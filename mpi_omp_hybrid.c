#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include "mpi.h"
#include<omp.h>
#include<unistd.h>
#include<string.h>

#define n 640
#define m 1024
#define iterations 500
#define PARM_CX 0.1
#define PARM_CY 0.1

int main( int argc, char* argv[]){

	MPI_Comm comm, old_comm;
	MPI_Group group;
	MPI_Request req1, req2, req3, req4, req5, req6, req7, req8, req9, req10, req11, req12, req13, req14, req15, req16;
	MPI_Status status;
	MPI_Datatype coltype, rowtype;

	int my_rank, comm_sz, i, j, I, N, M, sqrt_of_proc = 1, countReduce = 50, results, num_of_threads;

	double **ArrBefore, **ArrAfter, **tmpArr;
	time_t t;
	int north, east, south, west, rowPr = 1, colPr = 1, myrow, mycol, flag = 1, dim_size[2], periods[2], coords[2];
	double start_time, end_time, total_time;

	old_comm = MPI_COMM_WORLD;

	MPI_Init( &argc, &argv );
	MPI_Comm_size( old_comm, &comm_sz );
	MPI_Comm_rank( old_comm, &my_rank );
	
	//omp_set_num_threads(4);
	//num_of_threads = omp_get_num_threads();
	
	//xrhsimopoihsame metavlhth gia ton arithmo twn threads giati den allaze me thn entolh export OMP_NUM_THREADS
	num_of_threads = 8;
	if( my_rank == 0 ){	
		printf("Number of tasks : %d\n",comm_sz);
		printf("Threads per task : %d\n",num_of_threads);
	}

	/*******/
	//MPI_Comm_group(old_comm, &group);
	/*******/

	while( sqrt_of_proc * sqrt_of_proc < comm_sz ){
		sqrt_of_proc++;
	}
	if( sqrt_of_proc * sqrt_of_proc != comm_sz ){ //den uparxei h riza tou comm_sz
		if( comm_sz == 2 ){ //eidikh periptwsh me 2 processes
			dim_size[0] = 1;
			dim_size[1] = 2;
		}
		else{
			for(i = sqrt_of_proc; i < comm_sz; i++){
				if( comm_sz % i == 0 ){
					dim_size[0] = i;
					dim_size[1] = comm_sz / i;
					break;
				}
			}
			if( i == comm_sz ){
				if( my_rank == 0 ) printf("Wrong number of processes\n");
				exit(-1);
			}
		}
	}
	else{
		dim_size[0] = sqrt_of_proc;
		dim_size[1] = sqrt_of_proc;	
	}

	if( n % dim_size[0] != 0 || m % dim_size[1] != 0 ){
		printf("Wrong number of processes given\n");
		exit(-1);
	}

	N = n / dim_size[0];
	M = m / dim_size[1];

	periods[0] = 0;
	periods[1] = 0;

	//printf("My rank is %d\n", my_rank);

	/**  kathorismos geitonwn me kartesianh topologia **/
	MPI_Cart_create(old_comm, 2, dim_size, periods, 1, &comm);
	MPI_Cart_coords(comm, my_rank, 2, coords);
	//printf("Rank: %d, and coords: (%d, %d)\n", my_rank, coords[0], coords[1]);

	/* set column datatype */
	MPI_Type_vector( N, 1, M+2, MPI_DOUBLE, &coltype );
	MPI_Type_commit( &coltype );

	MPI_Cart_shift(comm, 0, 1, &north, &south);
	MPI_Cart_shift(comm, 1, 1, &west, &east);
	//printf("rank : %d --> north : %d, east : %d, south : %d, west : %d\n", my_rank, north, east, south, west);


	/* desmeush mnhmhs gia ton topiko pinaka ths diergasias */
	double *dataBef, *dataAft;
	ArrBefore = malloc( (N+2)*sizeof(*ArrBefore)+((N+2)*((M+2)*sizeof(**ArrBefore))) ); //kanoume ena malloc gia th desmeush twn didiastatwn pinakwn
	if( ArrBefore == NULL ){
		printf("Malloc failed on process %d\n",my_rank);
                fflush(stdout);
                exit(-1);
	}
	ArrAfter = malloc( (N+2)*sizeof(double*)+((N+2)*((M+2)*sizeof(double))) );
	if( ArrAfter == NULL ){
		printf("Malloc failed on process %d\n",my_rank);
                fflush(stdout);
                exit(-1);
	}
	dataBef = ArrBefore + N+2;
	dataAft = ArrAfter + N+2;
	for( i = 0; i < N+2; i++){
		ArrBefore[i] = dataBef + i*(M+2);
		ArrAfter[i] = dataAft + i*(M+2);
	}

	/* Gemisma pinaka me tuxaies times */
	srand( (unsigned) time(&t) );
	for( i = 1; i < N+1; i++){
		for( j = 1; j < M+1; j++){
			ArrBefore[i][j] = (double) ((rand() + my_rank) % 256);
			ArrAfter[i][j] = 0.0;
		}
	}

	//arxikopoihsh e3wterikwn shmeiwn se 0
	for(j = 0; j < M+2; j++){
		ArrBefore[0][j] = 0.0;
		ArrAfter[N+1][j] = 0.0;
	}

	for(i = 0; i < N+2; i++){
                ArrBefore[i][0] = 0.0;
                ArrAfter[i][M+1] = 0.0;
        }

	//ektupwsh pinakwn
	for(i = 1; i < N+1; i++){
		for(j = 1; j < M+1; j++){
			//printf("%6.1f ",ArrBefore[i][j]);
		}
		//printf("\n");
	}
	
	#pragma omp parallel num_threads(num_of_threads)
	start_time = MPI_Wtime();

	/**  KENTRIKO LOOP  **/	
	for( I = 0; I < iterations; I++ ){

		/***            KOMMATI PARALLHLIAS          ***/
		/* send */
		MPI_Isend( &ArrBefore[1][1], M, MPI_DOUBLE, north , 123, comm, &req1 );
		MPI_Isend( &ArrBefore[N][1], M, MPI_DOUBLE, south, 123, comm, &req2 );

		MPI_Isend( &ArrBefore[1][M], 1, coltype, east, 123, comm, &req3 );
		MPI_Isend( &ArrBefore[1][1], 1, coltype, west, 123, comm, &req4 );

		/* receive */
		MPI_Irecv( &ArrBefore[0][1], M, MPI_DOUBLE, north , 123, comm, &req9 );
		MPI_Irecv( &ArrBefore[N+1][1], M, MPI_DOUBLE, south, 123, comm, &req10 );

		MPI_Irecv( &ArrBefore[1][M+1], 1, coltype, east, 123, comm, &req11 );
		MPI_Irecv( &ArrBefore[1][0], 1, coltype, west, 123, comm, &req12 );


		/*Efarmogh filtrou seiriaka mono sta eswterika shmeia */
		//#pragma omp for schedule(static,1)
		#pragma omp for
		for( i = 2; i < N; i++){
			for(j = 2; j < M; j++){
				ArrAfter[i][j] = ArrBefore[i][j] + PARM_CX * ( ArrBefore[i+1][j] + ArrBefore[i-1][j] - 2.0 * ArrBefore[i][j] ) + PARM_CY * ( ArrBefore[i][j+1] + ArrBefore[i][j-1] - 2.0 * ArrBefore[i][j]);
                        }
		}

		/* wait gia ta recieve */
		MPI_Wait( &req9, &status);
        	MPI_Wait( &req10, &status);
        	MPI_Wait( &req11, &status);
        	MPI_Wait( &req12, &status);


		//e3wterikes seires
		//#pragma omp for schedule(static, 1)
		#pragma omp for
		for(j = 1; j < M+1; j++){
			ArrAfter[1][j] = ArrBefore[1][j] + PARM_CX * ( ArrBefore[2][j] + ArrBefore[0][j] - 2.0 * ArrBefore[1][j] ) + PARM_CY * ( ArrBefore[1][j+1] + ArrBefore[1][j-1] - 2.0 * ArrBefore[1][j]);
			ArrAfter[N][j] = ArrBefore[N][j] + PARM_CX * ( ArrBefore[N+1][j] + ArrBefore[N-1][j] - 2.0 * ArrBefore[N][j] ) + PARM_CY * ( ArrBefore[N][j+1] + ArrBefore[N][j-1] - 2.0 * ArrBefore[N][j]);
		}

		//e3wterikes sthles xwris ta gwniaka kelia
		//#pragma omp for schedule(static, 1)
		#pragma omp for
		for(i = 2; i < N; i++ ){
			ArrAfter[i][1] = ArrBefore[i][1] + PARM_CX * ( ArrBefore[i+1][1] + ArrBefore[i-1][1] - 2.0 * ArrBefore[i][1] ) + PARM_CY * ( ArrBefore[i][2] + ArrBefore[i][0] - 2.0 * ArrBefore[i][1]);
			ArrAfter[i][M] = ArrBefore[i][M] + PARM_CX * ( ArrBefore[i+1][M] + ArrBefore[i-1][M] - 2.0 * ArrBefore[i][M] ) + PARM_CY * ( ArrBefore[i][M+1] + ArrBefore[i][M-1] - 2.0 * ArrBefore[i][M]);
		}

                tmpArr = ArrBefore;
                ArrBefore = ArrAfter;
                ArrAfter = tmpArr;

		countReduce--;

		if(countReduce == 0){
			flag = 1; //idioi pinakes
			for(i = 1; i < N+1; i++){
				for(j = 1; j < M+1; j++){
					if(ArrBefore[i][j] != ArrAfter[i][j]){
						flag = 0;
						break;
					}
				}
				if(flag == 0) break;
			}

			results = 0;
			//reduce
			MPI_Allreduce( &flag, &results, 1, MPI_INT, MPI_SUM, comm);
			if( results == comm_sz ) printf("No changes on any process\n");
			countReduce = 50;
		}

		/* wait gia ta send */
		MPI_Wait( &req1, &status);
		MPI_Wait( &req2, &status);
		MPI_Wait( &req3, &status);
		MPI_Wait( &req4, &status);

		/*******End of calc*******/
	} //telos tou for

	end_time = MPI_Wtime();
	total_time = end_time - start_time;
	if( my_rank == 0 ){
		printf("Total time : %f\n", total_time);
	}

	free(ArrBefore);
	free(ArrAfter);

	MPI_Type_free( &coltype );
	MPI_Finalize();

}
