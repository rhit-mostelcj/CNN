import java.util.List;

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

    private List<int[][]> images;
    private List<Integer> labels;
    private double[][][] weightsC1 = new double[NUM_FEATURE_MAPS_C1][FILTER_SIZE_C1][FILTER_SIZE_C1];
    private double[][][][] weightsC3 = new double[NUM_FEATURE_MAPS_C3][NUM_FEATURE_MAPS_C1][FILTER_SIZE_C3][FILTER_SIZE_C3];
    private double[][][][] weightsC5 = new double[NUM_FEATURE_MAPS_C5][NUM_FEATURE_MAPS_C3][FILTER_SIZE_C5][FILTER_SIZE_C5];
    private double[][] weightsF6 = new double[NUM_UNITS_F6][NUM_FEATURE_MAPS_C5];
    private double[] biasesC1 = new double[NUM_FEATURE_MAPS_C1];
    private double[] biasesC3 = new double[NUM_FEATURE_MAPS_C3];
    private double[] biasesC5 = new double[NUM_FEATURE_MAPS_C5];
    private double[] biasesF6 = new double[NUM_UNITS_F6];


    public LeNet5(List<int[][]> images, List<Integer> labels) {
        this.images = images;
        this.labels = labels;
        initializeFiltersAndBiases();
    }

    private void initializeFiltersAndBiases() {
        for (int f = 0; f < NUM_FEATURE_MAPS_C1; f++) {
            for (int i = 0; i < FILTER_SIZE_C1; i++) {
                for (int j = 0; j < FILTER_SIZE_C1; j++) {
                    weightsC1[f][i][j] = Math.random() * 0.1 - 0.05;
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
    }

    public int[][] convertToBlackAndWhite(int[][] image) {
        for (int i = 0; i < image[0].length; i++) {
            for (int j = 0; j < image[0].length; j++) {
                if (image[i][j] > 0) {
                    image[i][j] = 1;
                }
            }
        }

        return image;
    }

    public int[][] addPadding(int[][] image) {
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

    public double[][][] convLayerC1(int[][] inputImage) {
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

    public double[][][] poolLayerS2(double[][][] inputFeatureMaps) {
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

    public double[][][] convLayerC3(double[][][] inputFeatureMaps) {
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

    public double[][][] poolLayerS4(double[][][] inputFeatureMaps) {
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

    public double[] convLayerC5(double[][][] inputFeatureMaps) {
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

    public double[] layerF6(double[] inputVector) {
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

}
