import java.util.Random;

public class DDPyjama {
    static Random rand = new Random(42);

    static String[] cannedLigands = 
        {"razvex", "qudgy", "afrs", "sst", "pgfht", "rt", 
        "id", "how", "aaddh",  "df", "os", "hid", 
        "sad", "fl", "rd", "edp", "dfgt", "spa"};

    // Ligand Score pair
    static class LSPair {
        String ligand;
        int score;
    
        public LSPair(String ligand, int score) {
            this.ligand = ligand;
            this.score = score;
        }

        @Override
        public String toString() {
            return "["+ligand+","+score+"]";
        }
    }

    // returns arbitrary string of lower-case letters of length at most max_ligand
    static String makeLigand(int maxLigandLength) {

        int len = rand.nextInt(maxLigandLength+1);
        if (len == 0) len++; // don't create a 0-character ligand
        
        StringBuilder sb = new StringBuilder();
        for (int i = 0;  i < len;  i++) 
            sb.append((char) ('a' + rand.nextInt(26)));  
        return sb.toString();
    }
  
    private static String[] generateLigands(int numLigands, int maxLigandLength, boolean useCanned) {
        // If we use canned ligands, use as many of them as we can, then fill the rest with randomly generated ligands. 
        // Otherwise, create a set of ligands whose length randomly varies from 1 to args.maxLigand

        String[] result = new String[numLigands];

        if (useCanned) {
            for(int i = 0; i < Math.min(numLigands, cannedLigands.length); i++) {
                result[i] = cannedLigands[i];
            }
        }

        for(int i = useCanned ? cannedLigands.length : 0; i < numLigands; i++) {
            result[i] = makeLigand(maxLigandLength);
        }
        return result;
    }

    public static void main(String[] args) {

        if (args.length != 4) {
            System.out.println("Usage DDPyjama numThreads numLigands maxLigandLength protein useCanned printLigands");

            // the example string below is one of Dijkstra's famous quotes
            System.out.println("   Example: java -cp .:Pyjama.jar DDPyjama 4 10 8 \"Simplicity is a great virtue but it requires hard work to achieve it and education to appreciate it\" false true\n");
        }

        int numThreads = 4;
        if (args.length >= 1) {
            numThreads = Integer.parseInt(args[0]);
        }

        int numLigands = 12;
        if (args.length >= 2) {
            numLigands = Integer.parseInt(args[1]);
        }

        int maxLigandLength = 6;
        if (args.length >= 3) {
            maxLigandLength = Integer.parseInt(args[2]);
        }

        String protein = "the cat in the hat wore the hat to the cat hat party";

        if (args.length >= 4) {
            protein = args[3];
        }

        System.out.println("Number of threads: " + numThreads);
        System.out.println("Number of ligands: "+numLigands);
        System.out.println("Max ligand length: "+ maxLigandLength);
        System.out.println("Protein: "+ protein);
        System.out.println();

        // Things to do: 
        // 1. Generate the requested numLigands w/ maxLigandLength
        // 2. Calculate the matching score for each ligand vs the given protein
        //    Score is calculated based on the number of character in the ligand that
        //    appears in the same order in the protein. 
        // 3. Find the ligand(s) with the highest score

        long start = System.currentTimeMillis();
        String[] ligands = generateLigands(numLigands, maxLigandLength, args.length >= 5 && args[4].equals("true"));

        // print the ligands if desired
        if (args.length >= 6 && args[5].equals("true")) {
            System.out.println("Here are the ligands");
            for(String l : ligands) {
                System.out.println(l);
            }
        }
        
        // map each ligand to (ligand, score)
        // also keep track of the maxScore 
        LSPair[] ligandsWScore = new LSPair[numLigands];
        int maxScore = 0;

        //#omp parallel for num_threads(numThreads) shared(numLigands, ligands, ligandsWScore, protein) reduction(max:maxScore) schedule(dynamic)
        for(int i = 0; i < numLigands; i++) {
            String ligand = ligands[i];
            int score = calcScore(ligand, protein);
            ligandsWScore[i] = new LSPair(ligand, score);
            maxScore = Math.max(maxScore, score);
        }

        // find the ligands whose score is maxScore
        // this is a reduce operation
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < numLigands; i++) {
            if (ligandsWScore[i].score == maxScore) {
                if (sb.length() > 0) sb.append(", ");
                sb.append(ligandsWScore[i].ligand);
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("The maximum score is " + maxScore);
        System.out.println("Achieved by ligand(s) "+ sb.toString());
        System.out.println("Calculation time " + (end-start) + " ms");
    }

    /**
     * Match a ligand (str1) and the protein. Count the number of characters in str1
     * that appear in the same seq in str2 (there can be any number of intervening chars)
     * @param str1 first string
     * @param str2 second string
     * @return number of matches
     */
    private static int calcScore(String str1, String str2) {
        // no match if either is empty string
        if (str1.length() == 0 || str2.length() == 0) return 0;

        if (str1.charAt(0) == str2.charAt(0)) {
            return 1 + calcScore(str1.substring(1), str2.substring(1));
        }
        return Math.max(
            calcScore(str1, str2.substring(1)), calcScore(str1.substring(1), str2));
    }
}