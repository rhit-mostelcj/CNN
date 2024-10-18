import java.util.List;

public class LeNet5 {

    private final int PADDING = 2;

    private List<int[][]> images;
    private List<Integer> labels;

    public LeNet5(List<int[][]> images, List<Integer> labels) {
        this.images = images;
        this.labels = labels;
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
