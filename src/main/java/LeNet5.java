import java.util.List;

public class LeNet5 {

    private static final int PADDING = 2;
    private static final int NUM_FILTERS_C1 = 6;
    private static final int FILTER_SIZE_C1 = 5;

    private List<int[][]> images;
    private List<Integer> labels;
    private double[][][] filtersC1 = new double[NUM_FILTERS_C1][FILTER_SIZE_C1][FILTER_SIZE_C1];
    private double[] biasesC1 = new double[NUM_FILTERS_C1];


    public LeNet5(List<int[][]> images, List<Integer> labels) {
        this.images = images;
        this.labels = labels;
        initializeFiltersAndBiases();
    }

    private void initializeFiltersAndBiases() {
        for (int f = 0; f < NUM_FILTERS_C1; f++) {
            for (int i = 0; i < FILTER_SIZE_C1; i++) {
                for (int j = 0; j < FILTER_SIZE_C1; j++) {
                    filtersC1[f][i][j] = Math.random() * 0.1 - 0.05;
                }
            }
            biasesC1[f] = Math.random() * 0.1 - 0.05;
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

}
