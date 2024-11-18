import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LeNet5 {
    private static final int PADDING = 2;
    private static final int NUM_FEATURE_MAPS_C1 = 6;
    private static final int FILTER_SIZE_C1 = 5;
    private static final int POOL_SIZE_S2 = 2;
    private static final int STRIDE_S2 = 2;
    private static final int NUM_FEATURE_MAPS_C3 = 16;
    private static final int FILTER_SIZE_C3 = 5;
    private static final int POOL_SIZE_S4 = 2;
    private static final int STRIDE_S4 = 2;
    private static final int NUM_FEATURE_MAPS_C5 = 120;
    private static final int FILTER_SIZE_C5 = 5;
    private static final int NUM_UNITS_F6 = 84;
    private static final int NUM_OUTPUT_CLASSES = 10;

    private double[][][] weightsC1 = new double[NUM_FEATURE_MAPS_C1][FILTER_SIZE_C1][FILTER_SIZE_C1];
    private double[][][][] weightsC3 = new double[NUM_FEATURE_MAPS_C3][NUM_FEATURE_MAPS_C1][FILTER_SIZE_C3][FILTER_SIZE_C3];
    private double[][][][] weightsC5 = new double[NUM_FEATURE_MAPS_C5][NUM_FEATURE_MAPS_C3][FILTER_SIZE_C5][FILTER_SIZE_C5];
    private double[][] weightsF6 = new double[NUM_UNITS_F6][NUM_FEATURE_MAPS_C5];
    private double[][] weightsOutput = new double[NUM_OUTPUT_CLASSES][NUM_UNITS_F6];
    private double[] biasesC1 = new double[NUM_FEATURE_MAPS_C1];
    private double[] biasesC3 = new double[NUM_FEATURE_MAPS_C3];
    private double[] biasesC5 = new double[NUM_FEATURE_MAPS_C5];
    private double[] biasesF6 = new double[NUM_UNITS_F6];

    private int[][] input;
    private double[][][] c1Output;
    private double[][][] s2Output;
    private double[][][] c3Output;
    private double[][][] s4Output;
    private double[] c5Output;
    private double[] f6Output;

    private double learningRate;

    public LeNet5() {
        initializeFiltersAndBiases();
        this.learningRate = 0.01;
    }

    private static int[] convertDoubleToInt(double[] doubleArray) {
        int rows = doubleArray.length;
        int[] intArray = new int[rows];

        for (int i = 0; i < rows; i++) {
                intArray[i] = (int) doubleArray[i];
        }

        return intArray;
    }

    private static int[][] convertDoubleToInt(double[][] doubleArray) {
        int rows = doubleArray.length;
        int cols = doubleArray[0].length;
        int[][] intArray = new int[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                intArray[i][j] = (int) doubleArray[i][j];
            }
        }

        return intArray;
    }

    private void initializeFiltersAndBiases() {
        Random random = new Random();

        for (int f = 0; f < NUM_FEATURE_MAPS_C1; f++) {
            for (int i = 0; i < FILTER_SIZE_C1; i++) {
                for (int j = 0; j < FILTER_SIZE_C1; j++) {
                    double std = Math.sqrt(2.0 / (FILTER_SIZE_C1 * FILTER_SIZE_C1));
                    weightsC1[f][i][j] = random.nextGaussian() * std;
                }
            }
            biasesC1[f] = Math.random() * 0.1 - 0.05;
        }

        for (int f = 0; f < NUM_FEATURE_MAPS_C3; f++) {
            for (int s = 0; s < NUM_FEATURE_MAPS_C1; s++) {
                for (int i = 0; i < FILTER_SIZE_C3; i++) {
                    for (int j = 0; j < FILTER_SIZE_C3; j++) {
                        weightsC3[f][s][i][j] = Math.random() * 0.1 - 0.05;
                    }
                }
            }
            biasesC3[f] = Math.random() * 0.1 - 0.05;
        }

        for (int f = 0; f < NUM_FEATURE_MAPS_C5; f++) {
            for (int s = 0; s < NUM_FEATURE_MAPS_C3; s++) {
                for (int i = 0; i < FILTER_SIZE_C5; i++) {
                    for (int j = 0; j < FILTER_SIZE_C5; j++) {
                        weightsC5[f][s][i][j] = Math.random() * 0.1 - 0.05;
                    }
                }
            }
            biasesC5[f] = Math.random() * 0.1 - 0.05;
        }

        for (int i = 0; i < NUM_UNITS_F6; i++) {
            for (int j = 0; j < NUM_FEATURE_MAPS_C5; j++) {
                weightsF6[i][j] = Math.random() * 0.1 - 0.05;
            }
            biasesF6[i] = Math.random() * 0.1 - 0.05;
        }

        for (int i = 0; i < NUM_OUTPUT_CLASSES; i++) {
            for (int j = 0; j < NUM_UNITS_F6; j++) {
                weightsOutput[i][j] = Math.random() * 0.1 - 0.05;
            }
        }
    }

    public void optimizeHyperParameters(int maxEpochs, List<int[][]> trainImages, List<Integer> trainLabels, List<int[][]> testImages, List<Integer> testLabels) {
        System.out.println("Optimizing epochs");
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            for (int i = 0; i < trainImages.size(); i++) {
                int[][] image = trainImages.get(i);
                int label = trainLabels.get(i);

                double[] predicted = forwardPass(image);
                backPropPass(predicted, label);
            }
            System.out.println("Epoch " + epoch + "/" + maxEpochs);
            double trainAccuracy = testNetwork(trainImages, trainLabels);
            System.out.println("Training accuracy: " + (Math.round(trainAccuracy * 10000.0) / 100.0) + "%");

            double testAccuracy = testNetwork(testImages, testLabels);
            System.out.println("Test accuracy: " + (Math.round(testAccuracy * 10000.0) / 100.0) + "%");
            System.out.println();
        }
        System.out.println("Done optimizing");
    }

    public void trainNetwork(int epochs, List<int[][]> trainImages, List<Integer> trainLabels) {
        System.out.println("Training started");
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = 0; i < trainImages.size(); i++) {
                int[][] image = trainImages.get(i);
                int label = trainLabels.get(i);

                double[] predicted = forwardPass(image);
                backPropPass(predicted, label);
            }
            System.out.println("Epoch " + epoch + "/" + epochs);

        }
        System.out.println("Training complete");
    }

    public double testNetwork(List<int[][]> testImages, List<Integer> testLabels) {
        int totalImages = testImages.size();
        int numCorrect = 0;

        for (int i = 0; i < totalImages; i++) {
            int[][] image = testImages.get(i);
            int label = testLabels.get(i);

            double[] outputs = forwardPass(image);
            int guess = getPrediction(outputs);

            if (guess == label) {
                numCorrect++;
            }
        }

        return (double) numCorrect / totalImages;
    }

    private int getPrediction(double[] outputs) {
        int index = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] > max) {
                max = outputs[i];
                index = i;
            }
        }

        return index;
    }

    public double[] forwardPassWithImage(int[][] image) {
        MNISTCNN.displayImage(image, "Original");
        input = addPadding(image);
        c1Output = convLayerC1(input);
        MNISTCNN.displayImage(convertDoubleToInt(c1Output[0]), "C1");
        s2Output = poolLayerS2(c1Output);
        MNISTCNN.displayImage(convertDoubleToInt(s2Output[0]), "S2");
        c3Output = convLayerC3(s2Output);
        MNISTCNN.displayImage(convertDoubleToInt(c3Output[0]), "C3");
        s4Output = poolLayerS4(c3Output);
        MNISTCNN.displayImage(convertDoubleToInt(s4Output[0]), "S4");
        c5Output = convLayerC5(s4Output);
        MNISTCNN.displayImage(convertDoubleToInt(c5Output), "C5");
        f6Output = layerF6(c5Output);
        MNISTCNN.displayImage(convertDoubleToInt(f6Output), "F6");
        double[] output = layerOutput(f6Output);
        MNISTCNN.displayImage(convertDoubleToInt(output), "Output");
        return output;
    }

    private double[] forwardPass(int[][] image) {
        input = addPadding(image);
        c1Output = convLayerC1(input);
        s2Output = poolLayerS2(c1Output);
        c3Output = convLayerC3(s2Output);
        s4Output = poolLayerS4(c3Output);
        c5Output = convLayerC5(s4Output);
        f6Output = layerF6(c5Output);
        return layerOutput(f6Output);
    }

    private void backPropPass(double[] predicted, int targetNumber) {
        double[] target = new double[NUM_OUTPUT_CLASSES];
        Arrays.fill(target, -1.0);
        target[targetNumber] = 1.0;

        double[] dOutput = computeOutputGradient(predicted, target);
        double[] dF6 = computeF6Gradient(dOutput);
        double[] dC5 = computeC5Gradient(dF6);
        double[][][] dS4 = computeS4Gradient(dC5);
        double[][][] dC3 = computeC3Gradient(dS4);
        double[][][] dS2 = computeS2Gradient(dC3);
        backpropagateToC1(dS2);
    }

    private double[] computeOutputGradient(double[] predicted, double[] target) {
        double[] dOutput = new double[NUM_OUTPUT_CLASSES];
        for (int i = 0; i < NUM_OUTPUT_CLASSES; i++) {
            dOutput[i] = 2 * (predicted[i] - target[i]) * tanhDerivative(predicted[i]);
        }

        return dOutput;
    }

    private double[] computeF6Gradient(double[] dOutput) {
        double[] dF6 = new double[NUM_UNITS_F6];
        for (int i = 0; i < NUM_UNITS_F6; i++) {
            double sum = 0.0;
            for (int j = 0; j < NUM_OUTPUT_CLASSES; j++) {
                sum += dOutput[j] * weightsOutput[j][i];
            }

            dF6[i] = sum * tanhDerivative(f6Output[i]);
        }

        for (int j = 0; j < NUM_OUTPUT_CLASSES; j++) {
            for (int i = 0; i < NUM_UNITS_F6; i++) {
                weightsOutput[j][i] -= learningRate * dOutput[j] * f6Output[i];
            }
        }

        return dF6;
    }

    private double[] computeC5Gradient(double[] dF6) {
        double[] dC5 = new double[NUM_FEATURE_MAPS_C5];
        for (int i = 0; i < NUM_FEATURE_MAPS_C5; i++) {
            double sum = 0.0;
            for (int j = 0; j < NUM_UNITS_F6; j++) {
                sum += dF6[j] * weightsF6[j][i];
            }

            dC5[i] = sum * tanhDerivative(c5Output[i]);
        }

        for (int j = 0; j < NUM_UNITS_F6; j++) {
            for (int i = 0; i < NUM_FEATURE_MAPS_C5; i++) {
                weightsF6[j][i] -= learningRate * dF6[j] * c5Output[i];
            }

            biasesF6[j] -= learningRate * dF6[j];
        }

        return dC5;
    }

    private double[][][] computeS4Gradient(double[] dC5) {
        int numFilters = s4Output.length;
        int xSize = s4Output[0].length;
        int ySize = s4Output[0][0].length;
        double[][][] dS4 = new double[numFilters][xSize][ySize];

        for (int f = 0; f < numFilters; f++) {
            for (int x = 0; x < xSize; x++) {
                for (int y = 0; y < ySize; y++) {
                    double sum = 0.0;
                    for (int i = 0; i < NUM_FEATURE_MAPS_C5; i++) {
                        sum += dC5[i] * weightsC5[i][f][x][y];
                    }

                    dS4[f][x][y] = sum * poolDerivative();
                }
            }
        }

        for (int i = 0; i < NUM_FEATURE_MAPS_C5; i++) {
            for (int f = 0; f < numFilters; f++) {
                for (int x = 0; x < xSize; x++) {
                    for (int y = 0; y < ySize; y++) {
                        weightsC5[i][f][x][y] -= learningRate * dC5[i] * s4Output[f][x][y];
                    }
                }
            }

            biasesC5[i] -= learningRate * dC5[i];
        }

        return dS4;
    }

    private double[][][] computeC3Gradient(double[][][] dS4) {
        int numFilters = c3Output.length;
        int featureSize = c3Output[0].length;
        double[][][] dC3 = new double[numFilters][featureSize][featureSize];

        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < featureSize; i++) {
                for (int j = 0; j < featureSize; j++) {
                    dC3[f][i][j] = dS4[f][i / 2][j / 2] * poolDerivative();
                }
            }
        }

        return dC3;
    }

    private double[][][] computeS2Gradient(double[][][] dC3) {
        int numFilters = s2Output.length;
        int featureSize = s2Output[0].length - FILTER_SIZE_C3 + 1;
        double[][][] dS2 = new double[numFilters][s2Output[0].length][s2Output[0].length];

        for (int f = 0; f < NUM_FEATURE_MAPS_C3; f++) {
            for (int i = 0; i < featureSize; i++) {
                for (int j = 0; j < featureSize; j++) {
                    for (int connectedMap : getC3Connectivity()[f]) {
                        for (int fi = 0; fi < FILTER_SIZE_C3; fi++) {
                            for (int fj = 0; fj < FILTER_SIZE_C3; fj++) {
                                dS2[connectedMap][i + fi][j + fj] += dC3[f][i][j] * weightsC3[f][connectedMap][fi][fj];
                            }
                        }
                    }
                }
            }
        }

        for (int f = 0; f < NUM_FEATURE_MAPS_C3; f++) {
            for (int i = 0; i < featureSize; i++) {
                for (int j = 0; j < featureSize; j++) {
                    for (int connectedMap : getC3Connectivity()[f]) {
                        for (int fi = 0; fi < FILTER_SIZE_C3; fi++) {
                            for (int fj = 0; fj < FILTER_SIZE_C3; fj++) {
                                weightsC3[f][connectedMap][fi][fj] -= learningRate * dC3[f][i][j] * s2Output[connectedMap][i + fi][j + fj];
                            }
                        }
                    }

                    biasesC3[f] -= learningRate * dC3[f][i][j];
                }
            }
        }

        return dS2;
    }

    private void backpropagateToC1(double[][][] dS2) {
        int featureSize = c1Output[0].length;

        double[][][] dC1 = new double[NUM_FEATURE_MAPS_C1][featureSize][featureSize];

        for (int f = 0; f < NUM_FEATURE_MAPS_C1; f++) {
            for (int i = 0; i < featureSize; i++) {
                for (int j = 0; j < featureSize; j++) {
                    dC1[f][i][j] = dS2[f][i / 2][j / 2] * poolDerivative();
                }
            }
        }

        for (int f = 0; f < NUM_FEATURE_MAPS_C1; f++) {
            for (int i = 0; i < featureSize; i++) {
                for (int j = 0; j < featureSize; j++) {
                    for (int fi = 0; fi < FILTER_SIZE_C1; fi++) {
                        for (int fj = 0; fj < FILTER_SIZE_C1; fj++) {
                            weightsC1[f][fi][fj] -= learningRate * dC1[f][i][j] * input[i + fi][j + fj];
                        }
                    }

                    biasesC1[f] -= learningRate * dC1[f][i][j];
                }
            }
        }
    }

    private int[][] convertToBlackAndWhite(int[][] image) {
        int[][] output = new int[image.length][image[0].length];

        for (int i = 0; i < image[0].length; i++) {
            for (int j = 0; j < image[0].length; j++) {
                if (image[i][j] > 0) {
                    output[i][j] = 1;
                } else {
                    output[i][j] = 0;
                }
            }
        }

        return output;
    }

    private int[][] addPadding(int[][] image) {
        int rows = image.length;
        int cols = image[0].length;

        int newRows = rows + 2 * PADDING;
        int newCols = cols + 2 * PADDING;

        int[][] newImage = new int[newRows][newCols];

        int paddingNum = 0;

        for (int i = 0; i < newRows; i++) {
            for (int j = 0; j < newCols; j++) {
                if ((i - PADDING) < 0 || (i + PADDING) >= newRows || (j - PADDING) < 0 || (j + PADDING) >= newCols) {
                    newImage[i][j] = paddingNum;
                } else {
                    newImage[i][j] = image[i - PADDING][j - PADDING];
                }
            }
        }

        return newImage;
    }

    private double activation(double a) {
        return Math.tanh(a);
    }

    private double tanhDerivative(double x) {
        return 1 - x * x;
    }

    private double poolDerivative() {
        return 1.0 / 4.0;
    }

    private double[][][] convLayerC1(int[][] inputImage) {
        int inputSize = inputImage.length;
        int outputSize = inputSize - FILTER_SIZE_C1 + 1;
        double[][][] outputFeatureMaps = new double[NUM_FEATURE_MAPS_C1][outputSize][outputSize];

        for (int f = 0; f < NUM_FEATURE_MAPS_C1; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double sum = 0.0;

                    for (int fi = 0; fi < FILTER_SIZE_C1; fi++) {
                        for (int fj = 0; fj < FILTER_SIZE_C1; fj++) {
                            sum += inputImage[i + fi][j + fj] * weightsC1[f][fi][fj];
                        }
                    }

                    sum += biasesC1[f];
                    outputFeatureMaps[f][i][j] = activation(sum);
                }
            }
        }

        return outputFeatureMaps;
    }

    private double[][][] poolLayerS2(double[][][] inputFeatureMaps) {
        int numFeatureMaps = inputFeatureMaps.length;
        int inputSize = inputFeatureMaps[0].length;
        int outputSize = inputSize / POOL_SIZE_S2;

        double[][][] outputFeatureMaps = new double[numFeatureMaps][outputSize][outputSize];

        for (int f = 0; f < numFeatureMaps; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double sum = 0.0;

                    for (int pi = 0; pi < POOL_SIZE_S2; pi++) {
                        for (int pj = 0; pj < POOL_SIZE_S2; pj++) {
                            sum += inputFeatureMaps[f][i * STRIDE_S2 + pi][j * STRIDE_S2 + pj];
                        }
                    }

                    outputFeatureMaps[f][i][j] = sum / (POOL_SIZE_S2 * POOL_SIZE_S2);
                }
            }
        }

        return outputFeatureMaps;
    }

    private int[][] getC3Connectivity() {
        return new int[][] {
                {0, 1, 2},
                {1, 2, 3},
                {2, 3, 4},
                {3, 4, 5},
                {4, 5, 0},
                {5, 0, 1}, // First 6
                {0, 1, 2, 3},
                {1, 2, 3, 4},
                {2, 3, 4, 5},
                {3, 4, 5, 0},
                {4, 5, 0, 1},
                {5, 0, 1, 2}, // Second 6
                {0, 1, 3, 4},
                {1, 2, 4, 5},
                {2, 3, 5, 0}, // Next 3
                {0, 1, 2, 3, 4, 5} // Last
        };
    }

    private double[][][] convLayerC3(double[][][] inputFeatureMaps) {
        int inputSize = inputFeatureMaps[0].length;
        int outputSize = inputSize - FILTER_SIZE_C3 + 1;
        double[][][] outputFeatureMaps = new double[NUM_FEATURE_MAPS_C3][outputSize][outputSize];

        int[][] connectivity = getC3Connectivity();

        for (int f = 0; f < NUM_FEATURE_MAPS_C3; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double sum = 0.0;

                    for (int connectedMap : connectivity[f]) {
                        for (int fi = 0; fi < FILTER_SIZE_C3; fi++) {
                            for (int fj = 0; fj < FILTER_SIZE_C3; fj++) {
                                sum += inputFeatureMaps[connectedMap][i + fi][j + fj] * weightsC3[f][connectedMap][fi][fj];
                            }
                        }
                    }

                    sum += biasesC3[f];

                    outputFeatureMaps[f][i][j] = activation(sum);
                }
            }
        }

        return outputFeatureMaps;
    }

    private double[][][] poolLayerS4(double[][][] inputFeatureMaps) {
        int numFeatureMaps = inputFeatureMaps.length;
        int inputSize = inputFeatureMaps[0].length;
        int outputSize = inputSize / POOL_SIZE_S4;

        double[][][] outputFeatureMaps = new double[numFeatureMaps][outputSize][outputSize];

        for (int f = 0; f < numFeatureMaps; f++) {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    double sum = 0.0;

                    for (int pi = 0; pi < POOL_SIZE_S4; pi++) {
                        for (int pj = 0; pj < POOL_SIZE_S4; pj++) {
                            sum += inputFeatureMaps[f][i * STRIDE_S4 + pi][j * STRIDE_S4 + pj];
                        }
                    }

                    outputFeatureMaps[f][i][j] = sum / (POOL_SIZE_S4 * POOL_SIZE_S4);
                }
            }
        }

        return outputFeatureMaps;
    }

    private double[] convLayerC5(double[][][] inputFeatureMaps) {
        double[] outputFeatureMaps = new double[NUM_FEATURE_MAPS_C5];

        for (int f = 0; f < NUM_FEATURE_MAPS_C5; f++) {
            double sum = 0.0;

            for (int s = 0; s < NUM_FEATURE_MAPS_C3; s++) {
                for (int fi = 0; fi < FILTER_SIZE_C5; fi++) {
                    for (int fj = 0; fj < FILTER_SIZE_C5; fj++) {
                        sum += inputFeatureMaps[s][fi][fj] * weightsC5[f][s][fi][fj];
                    }
                }
            }

            sum += biasesC5[f];

            outputFeatureMaps[f] = activation(sum);
        }

        return outputFeatureMaps;
    }

    private double[] layerF6(double[] inputVector) {
        double[] outputVector = new double[NUM_UNITS_F6];

        for (int i = 0; i < NUM_UNITS_F6; i++) {
            double sum = 0.0;

            for (int j = 0; j < NUM_FEATURE_MAPS_C5; j++) {
                sum += weightsF6[i][j] * inputVector[j];
            }
            sum += biasesF6[i];

            outputVector[i] = activation(sum);
        }

        return outputVector;
    }

    private double[] layerOutput(double[] inputVector) {
        int inputSize = inputVector.length;
        double[] outputVector = new double[NUM_OUTPUT_CLASSES];

        for (int i = 0; i < NUM_OUTPUT_CLASSES; i++) {
            double sum = 0.0;

            for (int j = 0; j < inputSize; j++) {
                sum += inputVector[j] * weightsOutput[i][j];
            }
            outputVector[i] = activation(sum);
        }

        return outputVector;
    }
}
