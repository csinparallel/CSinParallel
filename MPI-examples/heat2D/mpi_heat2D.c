/****************************************************************************
 * FILE: mpi_heat2D.c
 * OTHER FILES: draw_heat.c
 * DESCRIPTIONS:
 *   HEAT2D Example - Parallelized C Version
 *   This example is based on a simplified two-dimensional heat
 *   equation domain decomposition.  The initial temperature is computed to be
 *   high in the middle of the domain and zero at the boundaries.  The
 *   boundaries are held at zero throughout the simulation.  During the
 *   time-stepping, an array containing two domains is used; these domains
 *   alternate between old data and new data.
 *
 *   In this parallelized version, the grid is decomposed by the conductor
 *   process and then distributed by rows to the worker processes.  At each
 *   time step, worker processes must exchange border data with neighbors,
 *   because a grid point's current temperature depends upon it's previous
 *   time step value plus the values of the neighboring grid points.  Upon
 *   completion of all time steps, the worker processes return their results
 *   to the conductor process.
 *
 *   Two data files are produced: an initial data set and a final data set.
 *   An X graphic of these two states displays after all calculations have
 *   completed.
 * AUTHOR: Blaise Barney - adapted from D. Turner's serial C version. Converted
 *   to MPI: George L. Gusciora (1/95)
 * LAST REVISED: 06/12/13 Blaise Barney
 * LAST REVISED by  Libby Shoop   12/23  Extensive changes for educational
 *                                       purposes and scaling to large plates.
 ****************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "display.h"
#include "getCommandLine.h"

float inidat();    // data initialization
void prtdat();     // for debugging, print the cell values
void update();     // update heat values at each time step

#define NXPROB 512   /* x dimension of problem grid */
#define NYPROB 512   /* y dimension of problem grid */
#define STEPS 200    /* number of time steps */
#define MAXWORKER 16 /* maximum number of worker tasks */
#define MINWORKER 1  /* minimum number of worker tasks */
#define BEGIN 1      /* message tag */
#define LTAG 2       /* message tag */
#define RTAG 3       /* message tag */
#define DONE 4       /* message tag */
#define NONE 0       /* indicates no neighbor */
#define CONDUCTOR 0  /* taskid of first process */

// values used to update the heat value at each cell for each timestep.
// see the update() function below.
struct Parms
{
  float cx;
  float cy;
} parms = {0.1, 0.1};  // increasing makes heat dissipate faster

int main(int argc, char *argv[])
{

  int taskid,                      /* this task's unique id */
      numworkers,                  /* number of worker processes */
      numtasks,                    /* number of tasks/processes */
      averow, rows, offset, extra, /* for sending rows of data */
      dest, source,                /* to - from for message send-receive */
      left, right,                 /* neighbor tasks */
      msgtype,                     /* for message types */
      rc,                          /*  return code from abort*/
      start, end,                  /* rows of work for each process */
      i, ix, iy, iz, it;           /* loop variables */
  MPI_Status status;

  double startTime = 0.0, totalTime = 0.0; // for timing

  int nx, ny, steps; /* x,y dim of grid and number of steps*/
  int display = 0;   // choose to display the start and end grid
  int verbose = 0;   // print grid to std out instead

  rc = -1; // return when something went wrong and aborted
  
  MPI_Init(&argc, &argv);

  /* First, find out my taskid and how many tasks are running */
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  numworkers = numtasks - 1;

  // initial default grid dimensions
  nx = NXPROB;  // number of rows
  ny = NYPROB;  // number of columns

  steps = STEPS;  // number of 'time' steps

  float max_temp = 0.0;  // for use in plot coloring

  getArguments(argc, argv, &nx, &ny, &steps, &display, &verbose);

 

  float u[2][nx][ny]; /* 3D array for two states of the grid */
  // NOTE: throughout this code, visualize like this:
  //       x is used for rows, starting at 0 going down and
  //       y is used for columns, starting at 0 going right
  //       the 2 indicates that we will hold 2 versions of the
  //       grid of cells with nx rows and ny columns.

   // all wait; conductor starts timing 
  MPI_Barrier(MPI_COMM_WORLD);
  if (taskid == CONDUCTOR) {
    startTime = MPI_Wtime();
  }

  if (taskid == CONDUCTOR)
  {
    /************************* conductor code *******************************/
    /* Check if numworkers is within range - quit if not */
    if ((numworkers > MAXWORKER) || (numworkers < MINWORKER))
    {
      printf("ERROR: the number of tasks must be between %d and %d.\n",
             MINWORKER + 1, MAXWORKER + 1);
      printf("Quitting...\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
      exit(1);
    }

     // don't print to screen if too large
    if (verbose && (nx > 20 || ny > 20)) {
      verbose = 0;
      printf("Warning: nx or ny was too large for printing grids.\n");
      printf("         Turning off verbose mode in conductor.\n");
    }

    printf("Starting mpi_heat2D with %d worker tasks.\n", numworkers);

    /* Initialize grid */
    printf("Grid size: X= %d  Y= %d  Time steps= %d\n", nx, ny, steps);
    printf("Initializing grid ...\n");
    max_temp = inidat(nx, ny, u);

    if (display)
    {
      printf("writing initial.dat file\n");
      prtdat(nx, ny, u, "initial.dat");
    } 
  
    if (verbose)
    {
      printf("max value in center: %f\n", max_temp);
      printf("start grid values:\n");
      prtdat(nx, ny, u, NULL);
    }

    /* Distribute work to workers.  Must first figure out how many rows to */
    /* send and what to do with extra rows.  */
    averow = nx / numworkers;
    extra = nx % numworkers;
    if (verbose)
    {
      printf("averow= %d, extra= %d\n", averow, extra);
    }
    offset = 0;
    // conductor goes through each worker
    for (i = 1; i <= numworkers; i++)
    {
      rows = (i <= extra) ? averow + 1 : averow;
      /* Tell each worker who its neighbors are, since they must exchange */
      /* data with each other. */
      if (i == 1) {     
        left = NONE;
      } else {
        left = i - 1;
      }
      if (i == numworkers) {
        right = NONE;
      } else {
        right = i + 1;
      }
      /*  Now send startup information to each worker  with BEGIN tag*/
      dest = i;
      MPI_Send(&offset, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);  // start row
      MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);    // numrows
      MPI_Send(&left, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);    // neighbor process
      MPI_Send(&right, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);   // neighbor process
      MPI_Send(&u[0][offset][0], rows * ny, MPI_FLOAT, dest, 
               BEGIN, MPI_COMM_WORLD);                             // data in grid slice
      
      // debug:
      if (verbose) {
        printf("Sent to task %d: rows= %d offset= %d ",dest,rows,offset);
        printf("left= %d right= %d\n",left,right);
      }
      offset = offset + rows;
      
    }
    /* Now wait for results from all worker tasks */
    for (i = 1; i <= numworkers; i++)
    {
      source = i;
      msgtype = DONE;                    // tagged 'DONE'
      MPI_Recv(&offset, 1, MPI_INT, source, msgtype, 
               MPI_COMM_WORLD, &status); // offset row
      MPI_Recv(&rows, 1, MPI_INT, source, msgtype, 
               MPI_COMM_WORLD, &status); // num rows
      MPI_Recv(&u[0][offset][0], rows * ny, MPI_FLOAT, 
               source, msgtype, MPI_COMM_WORLD, &status); // data in grid slice
    }

    /* Write final output, call X graph and finalize MPI */
    // printf("Writing final.dat file and generating graph...\n");
    if (display)
    {
      printf("Writing final.dat file ...\n");
      prtdat(nx, ny, &u[0][0][0], "final.dat");
    }

    if (verbose)
    {
      printf("end grid values:\n");
      prtdat(nx, ny, u, NULL);
    }
    
  } /* End of conductor code */

  /************************* workers code ******************************/
  if (taskid != CONDUCTOR)
  {
    /* Initialize everything - including the borders - to zero */
    for (iz = 0; iz < 2; iz++)
      for (ix = 0; ix < nx; ix++)
        for (iy = 0; iy < ny; iy++)
          u[iz][ix][iy] = 0.0;

    /* Receive my offset, rows, neighbors and grid partition from conductor */
    source = CONDUCTOR;
    msgtype = BEGIN;
    MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&left, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&right, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&u[0][offset][0], rows * ny, MPI_FLOAT, source, msgtype,
             MPI_COMM_WORLD, &status);

    /* Determine border elements.  Need to consider first and last rows. */
    /* Obviously, row 0 can't exchange with row 0-1.  Likewise, the last */
    /* row can't exchange with last+1.  */
    start = offset;
    end = offset + rows - 1;
    if (offset == 0)
      start = 1;
    if ((offset + rows) == nx)
      end--;
    
    if (verbose) {
      printf("task=%d  start=%d  end=%d\n",taskid,start,end);
      printf("Task %d received work. Beginning time steps...\n",taskid);
    }

    /* Begin doing steps iterations.  Must communicate border rows with */
    /* neighbors.  If I have the first or last grid row, then I only need */
    /*  to  communicate with one neighbor  */ 
    iz = 0; // old, new grid swaps between u[0][][] and u[1][][]
    for (it = 1; it <= steps; it++)
    {
      if (left != NONE)
      {
        MPI_Send(&u[iz][offset][0], ny, MPI_FLOAT, left,
                 RTAG, MPI_COMM_WORLD);
        source = left;
        msgtype = LTAG;
        MPI_Recv(&u[iz][offset - 1][0], ny, MPI_FLOAT, source,
                 msgtype, MPI_COMM_WORLD, &status);
      }
      if (right != NONE)
      {
        MPI_Send(&u[iz][offset + rows - 1][0], ny, MPI_FLOAT, right,
                 LTAG, MPI_COMM_WORLD);
        source = right;
        msgtype = RTAG;
        MPI_Recv(&u[iz][offset + rows][0], ny, MPI_FLOAT, source, msgtype,
                 MPI_COMM_WORLD, &status);
      }

      /* Now call update to update the value of grid points */
      // old, new grid swaps between u[0][][] and u[1][][]
      update(start, end, ny, &u[iz][0][0], &u[1 - iz][0][0]);

      iz = 1 - iz;  // if 0, becomes 1; if 1, becomes 0
    } // end of steps

    /* Finally, send my portion of final results back to conductor */
    MPI_Send(&offset, 1, MPI_INT, CONDUCTOR, DONE, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, CONDUCTOR, DONE, MPI_COMM_WORLD);
    MPI_Send(&u[iz][offset][0], rows * ny, MPI_FLOAT, CONDUCTOR, DONE,
             MPI_COMM_WORLD);
    
  }
  /******************** end  workers code ******************************/

  MPI_Barrier(MPI_COMM_WORLD); // wait for all to get here for timing
  if (taskid == CONDUCTOR)
  {
    totalTime = MPI_Wtime() - startTime;
    printf("\nTime: %f secs.\n\n", totalTime);
    if (display)
    {
      float ratio = (float)ny / (float)nx;  // y axis to x axis
      draw2DHeat(max_temp, ratio);
      printf("Pausing to display plots. Press Enter to finish.");
      fflush(stdout);
      getchar();
    }
    
  }

  MPI_Finalize();
  return 0;
}

/**************************************************************************
 *  subroutine update
 * params:
 *   start = starting row
 *   end = ending row
 *   nc = number of columns
 *   u1 = start of source grid
 *   u2 = start of destination grid for new values
 ****************************************************************************/
void update(int start, int end, int nc, float *u1, float *u2)
{
  int ir, ic;      // row, column counters
  int north_row;   // in loop, row above, or north
  int south_row;   // in loop, row below, or south
  int east_col;    // in loop, column to right, or east
  int west_col;    // in loop, column to left, or west

  for (ir = start; ir <= end; ir++) {
    south_row = ir + 1;
    north_row = ir - 1;
  
    for (ic = 1; ic <= nc - 2; ic++) {
      east_col = ic + 1;
      west_col = ic - 1;

      // compute new value for u2 grid:
      //
      // cell at ir, ic in u2 grid becomes
      // cell at ir, ic in u1 grid plus 
      //     param in n-s direction times 
      //        (cell value to south + cell value to north -
      //         2 * cell at ir, ic)
      //     plus
      //     param in e-w direction times
      //        (cell value to the east + cell value to the west -
      //         2 * cell at ir, ic)

      *(u2 + ir * nc + ic) = *(u1 + ir * nc + ic) +
                             parms.cx * (*(u1 + (south_row) * nc + ic) +
                                         *(u1 + (north_row) * nc + ic) -
                                         2.0 * *(u1 + ir * nc + ic)) +
                             parms.cy * (*(u1 + ir * nc + east_col) +
                                         *(u1 + ir * nc + west_col) -
                                         2.0 * *(u1 + ir * nc + ic));


    }
  }
}

/**************************************************************
 *  subroutine inidat
 * 
 * Updated by Libby Shoop to enable simulation of larger plates
 * with a heat source of concentric circles in the center.
 **************************************************************/
float inidat(int nx, int ny, float *u)
{
  int ix, iy;
  float max = 0.0;
  float next = 0.0; // next temp
 
  float xc = (float)nx/2;
  float yc = (float)ny/2;
  float r1 = (float)nx * 0.15;
  float r2 = (float)nx * 0.25;
  float r3 = (float)nx * 0.35;

  float sum_squares = 0.0;
  float inner_temp = 3600.0;

  for (ix = 0; ix <= nx - 1; ix++) {
    for (iy = 0; iy <= ny - 1; iy++) {
      // this original code only worked for a very small nx, ny
      //next = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));

      // In this version, we set a temp based on where in a 
      // circle on the plate a point lies, with zero temp 
      // around the outside.
      // For any point ix, iy and center xc, yc, 
      // the sum of the squares of (ix - xc) and (iy -yc)
      // is equal to the square of the radius.
      // So we will set temps based on whether the point
      // is inside one of three concentric circles or not.
      sum_squares = ((ix - xc) * (ix -xc)) + ((iy - yc) * (iy - yc));

      if ( sum_squares < (r1 * r1) ) {
        next = inner_temp;
      } else if ( sum_squares < (r2 * r2) ) {
        next = inner_temp * 0.8;
      } else if ( sum_squares < (r3 * r3) ) {
        next = inner_temp * 0.6;
      } else {
        next = 0.0;
      }
     
      *(u + ix * ny + iy) = next;
      max = fmaxf(max, next);
    }
  }
  return max;
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam)
{
  int ix, iy;
  FILE *fp;

  if (fnam == NULL) {
    fp = stdout;
  } else {
    fp = fopen(fnam, "w");
  }

  // prints to convert to decreasing y, 
  // increasing x for plotting
  for (iy = ny - 1; iy >= 0; iy--)
  {
    for (ix = 0; ix <= nx - 1; ix++)
    {
      fprintf(fp, "%8.1f", *(u1 + ix * ny + iy));
      if (ix != nx - 1)
        fprintf(fp, " ");
      else
        fprintf(fp, "\n");
    }
  }
  if (fnam != NULL)
    fclose(fp);
}

