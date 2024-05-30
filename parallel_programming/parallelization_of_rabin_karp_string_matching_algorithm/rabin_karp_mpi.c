
/*
version: 2, Date: 3-Aug-2022, Group: 6
Parameters:
      Change parameters pattern, filePath, splitNum as required
      under section "Assign Variable Values" inside main() method.
      Specify number of threads and processes from command.
Compile and Run:
      Windows:
            $ gcc rabin_karp_mpi.c -o app.exe -l msmpi -L "C:\msmpisdk\Lib\x64" -I "C:\msmpisdk\Include"
            $ mpiexec -n 16 app.exe
      Linux:
            $ mpicc rabin_karp_mpi.c -o app.exe
            $ mpirun -n 16 ./app.exe       
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

/*
Rabin-Karp search function with rolling hash
*/
int rabin_karp_search(char pattern[], char text[], int base, int div) {

  int i, j; // iteration index

  int m = strlen(pattern);
  int n = strlen(text);
  int c = 0; // counter for number of matches

  int ph = 0; //pattern hash
  int th = 0; // text hash
  int bh = 1; // base hash

  // Get hash value of bases of any (m-1)-length-text. Used in rolling hash calculation
  for (i = 0; i < m - 1; i++)
    bh = (bh * base) % div;

  // Get hash value for pattern and m-length-text at the beginning of the text
  for (i = 0; i < m; i++) {
    ph = (base * ph + pattern[i]) % div;
    th = (base * th + text[i]) % div;
  }

  // Find the match
  for (i = 0; i <= n - m; i++) {//loop1 START
    if (ph == th) {
      for (j = 0; j < m; j++) {
        if (text[i + j] != pattern[j])
          break;
      }

      if (j == m){//j will be equal to m if all characters match
        // printf("Match at position:  %d \n", i + 1);
        c++;
      }

    }

    // Rolling hash calculation
    if (i < n - m) {
      th = (base * (th - text[i] * bh) + text[i + m]) % div;
      if (th < 0)
        th = (th + div);
    }
  }//loop1 END

  // printf("Total Match %d", c);
  return c;
}


/*
Format and print size in bytes, kb, mb or gb
depending on the given bytes (nb)
*/
void print_size(size_t nb, char *prefix){
  char *unit[4] = {"bytes", "kb", "mb", "gb"};
  double *result = (double*)calloc(4,sizeof(double));
  result[0]=nb;
  int i;
  for(i=1;i<4;i++){
    result[i] = result[i-1]/1024;
  }

  for(i=3; i>=0; i--){
    if(result[i]>1){
      printf("%s%.2f %s", prefix,result[i], unit[i]);
      break;
    }
  }
  free(result);
}

/* 
Parallel MPI IO is used to read chunk of files by each process. Each 
split is overlapped with one of the other splits by n-1 character
where n is the length of the search pattern. The reason for this 
overlapping is explained in the report.
 */
 
int main(int argc, char* argv[])
{	
  // Timer: start time
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);

  // =================== Assign Variable Values ==============
  char pattern[] = "passed";
  char filePath[] = "./data/windows_mb512.log";
  // int splitNum = 16; // This is replaced by nprocs
  int base = 10;
  int div = 17;
  // =========================================================

  int k = strlen(pattern);

    MPI_Init(&argc, &argv);

    // Process rank in the communicator
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
    // Number of processes
	int nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;
    MPI_Offset filesize;

    MPI_File_open(MPI_COMM_WORLD, filePath, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_File_get_size (fh, &filesize); /* in bytes */

    size_t bufsize = filesize/nprocs;

    offset = rank*bufsize;//

    char *buf;
    if(rank == nprocs-1){
        bufsize = filesize - rank*bufsize;
    }
    buf = (char *) malloc(bufsize+(k-1));

    //
    MPI_File_read_at(fh, offset, buf, bufsize+(k-1), MPI_CHAR, &status);

    int result_local = rabin_karp_search(pattern, buf, base, div);//search by rabin-karp

    printf("\n======= Process %d Count %d =======\n",rank, result_local);


    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        int result[nprocs];
        int total=0;
        MPI_Gather(&result_local, 1, MPI_INT, result, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int i;

        for(i=0; i < nprocs; i++){
            total = total + result[i];
        }
        printf("\n________________________________________");
        printf("\n Total Count: %d", total);
        // Timer: end time
        gettimeofday(&tv2, NULL);
        printf ("\n Total Time: %.3f seconds",
        (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
        (double) (tv2.tv_sec - tv1.tv_sec));
        
        printf("\n Search Pattern: %s", pattern);
        printf("\n Text File: %s", filePath);
        printf("\n");
        print_size(filesize, " File Size: ");
        printf("\n________________________________________\n");
        printf("\n");
    }
    else
    {
        MPI_Gather(&result_local, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(buf);
    MPI_File_close(&fh);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
