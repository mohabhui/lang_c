/*
version: 2, Date: 3-Aug-2022, Group: 6
Parameters:
      Change parameters pattern, filePath, splitNum as required
      under section "Assign Variable Values" inside main() method.
      Specify number of threads and processes from command.
Compile and Run:
      Windows:
            $ gcc -DNTHREADS=16 -fopenmp rabin_karp_omp.c -o app.exe
            $ set OMP_NUM_THREADS=16
            $ app.exe
      Linux:
            $ gcc -DNTHREADS=16 -fopenmp rabin_karp_omp.c -o app.exe
            $ export OMP_NUM_THREADS=16
            $ ./app.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

/*
File stream and total number of bytes are stored in this
struct for subsequent use. Once we create this struct, we
need not to open and close the file for all processes or threads
in parallel programming. Remember to close the file stream (*fs)
after all operations.
*/
typedef struct fileData{
  FILE *fs;// file stream
  long nb; // number of bytes
} fdata;



/*
Make fdata struct and return it.
Open a file and assign the file stream to *fs, get the file size in bytes and
assign it to nb.
*/
fdata load_fileData(char filePath[], fdata fd){
  FILE    *fs;
  fs = fopen(filePath, "r");

  long nb; // number of bytes
  fseek(fs, 0L, SEEK_END);
  nb = ftell(fs);
  fseek(fs, 0L, SEEK_SET);

  fd.fs = fs;
  fd.nb = nb;

  // if fs is closed in this routine, fs of fdata will not be available;
  // Don't forget to close it in your app.
  // fclose(fs);

  return fd;
}


/*
Read file stream from supplied fdata. Able to read the whole file stream or
specified chunk of the file stream.

We can specify number of splits by totalChunks and get the desired chunk by idxChunk arguments.

OverlapChars argument is used if we split large file stream into multiple streams
and we search the pattern in the streams in parallel. If the pattern length is n characters
long, we need to set value of overlapChars to n-1, so that pattern in the file streams
that are splitted, can not be missed, e.g., we search a pattern EFG in a stream of
ABCDEFGHIJ; the stream is splitted into 3 processes as ABC|DEF|GHIJ. None of the
processes will find the pattern EFG. If we overlap 2 characters (3-1 = 2) at the
end of each stream except the last one, then the splits will be ABCDE|DEFGH|GHIJ and
the pattern EFG will be found.

To read the whole file/stream, value of totalChunks is 1. idxChunk and overlapChars
arguments are not considered  by the function and can be set to 0 and 0.

To split the file/stream into n chunks without overlapping the chunk and get the
chunk at index i, values of totalChunks, idxChunk and overlapChars will be n, i and 0.

*/
char *read_fileData(fdata fd, int totalChunks, int idxChunk, int overlapChars){

    // shorten the var name for convenience
    FILE *fs;
    long nb;
    fs = fd.fs;
    nb = fd.nb;

    int n = totalChunks;

    char *txt;

    if(n == 1){
      fseek(fs, 0L, SEEK_SET);
      txt = (char*)calloc(nb, sizeof(char));
      fread(txt, sizeof(char), nb, fs);
    }else{
      // shorten the var name for convenience
      int i = idxChunk;
      int olc = overlapChars;

      long cs1; //cs1 is size of each chunks except last chunk
      long cs2; // cs2 is size of last chunks
      cs1 = nb/n;
      cs2 = nb-(nb/n)*(n-1);

      fseek(fs, cs1*i, SEEK_SET);
      if(i<n-1){
        txt = (char*)calloc(cs1+olc, sizeof(char));
        fread(txt, sizeof(char), cs1+olc, fs);
      }else if(i==n-1){
        txt = (char*)calloc(cs2, sizeof(char));
        fread(txt, sizeof(char), cs2, fs);
      }else{
        ;
      }
    }

    // printf("%s\n", txt);
    return txt;
}

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


int main() {

  int numthreads=NTHREADS;
  int result[numthreads];


  // Timer: start time
  struct timeval  tv1, tv2;
  gettimeofday(&tv1, NULL);

  // =================== Assign Variable Values ==============
  char pattern[] = "passed";
  char filePath[] = "./data/windows_mb512.log";
  // int splitNum = 16; // This is replaced by numthreads
  int base = 10;
  int div = 17;
  // =========================================================

  fdata md;// md->mydata
  md = load_fileData(filePath, md);
  int k = strlen(pattern);

  int j;//for iteration
  int total=0;


  #pragma  omp  parallel shared(result, total)
    {     int result_local;
          char* mytext;
          int ID;
          ID = omp_get_thread_num();
          #pragma omp critical
          {
            mytext = read_fileData(md,numthreads, ID, k-1);//overlap-split the file stream and read the chunk at index equal to thread ID
          }
          result_local = rabin_karp_search(pattern, mytext, base, div);//search by rabin-karp

            result[ID] = result_local;

            printf("\n======= Thread %d Count %d =======",ID, result[ID]);

            total=total+result_local;

          // printf("\n%s\n", mytext);
          free(mytext);//free the memory allocated in read_fileData function

    }// end of #pragma

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
        print_size(md.nb, " File Size: ");
        printf("\n________________________________________\n");
        printf("\n");

  fclose(md.fs);//free file stream of fileData struct
}
