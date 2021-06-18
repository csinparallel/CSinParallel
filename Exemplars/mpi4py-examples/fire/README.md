# Simulating Forest Fires Using Python and MPI

This code is the same as what we have provided for the Raspberry Pi clusters and explained in [this section of a book explaining its use](https://pdcbook.calvin.edu/pdcbook/RaspberryPi-mpi4py/09Exemplars/forest_fire.html).

To work through these examples from a remote server, it will be necessary to be able to remotely connect to a desktop on it, using VNC. If you cannot do that, these examples would need to change to eliminate the visual displays of results that are given in the code.

## Try some simulations

 On a server with 16, 32, or 64 cores available, you will be able to try some larger examples than what you used on the Pi clusters. You also do not need a 'hostfile' when using MPI on a single multiprocessor machine.

 Here is an example for a forest size of 30x30 trees, the probability threshold is incremented by 0.1 from 0 to 1.0, and the number of trials is 40.

    mpirun -np 4 python fire_mpi_simulate.py 30 0.1 40

Be patient as the simulation runs. It should display the resulting graphs when it completes after 20 seconds or so on a server-class processor.

Try running these tests:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-0pky">-np</th>
    <th class="tg-0pky">tree row size</th>
    <th class="tg-0pky">probability increment</th>
    <th class="tg-0pky">number of trials</th>
    <th class="tg-0pky">running time</th>
  </tr>
  <tr>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">0.1</td>
    <td class="tg-0pky">40</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">8</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">0.1</td>
    <td class="tg-0pky">40</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">16</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">0.1</td>
    <td class="tg-0pky">40</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">0.1</td>
    <td class="tg-0pky">40</td>
    <td class="tg-0pky"></td>
  </tr>
</table>

What do you observe about the time as you double the number of workers?

When does the message passing cause the most overhead, which adds to the running time?

You could try increasing the number of trials to 80 and comparing 8 to 16 to 32 processes.

Note how the time spent for each process to do its work varies. Can you explain why this is?
