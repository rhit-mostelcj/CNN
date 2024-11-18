import javax.swing.*;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MNISTCNN {

    public static void main(String[] args) {
        String trainImagesPath = "train-images-idx3-ubyte";
        String trainLabelsPath = "train-labels-idx1-ubyte";
        String testImagesPath = "t10k-images-idx3-ubyte";
        String testLabelsPath = "t10k-labels-idx1-ubyte";
        try {
            System.out.println("Reading in training files");
            List<int[][]> trainImages = readImages(trainImagesPath);
            List<Integer> trainLabels = readLabels(trainLabelsPath);
            System.out.println("Done reading training files");

            LeNet5 leNet5 = new LeNet5();
            int epochs = 1;
            leNet5.trainNetwork(epochs, trainImages, trainLabels);

            System.out.println("Reading in test files");
            List<int[][]> testImages = readImages(testImagesPath);
            List<Integer> testLabels = readLabels(testLabelsPath);
            System.out.println("Done reading test files");

            double trainAccuracy = leNet5.testNetwork(trainImages, trainLabels);
            System.out.println("Training accuracy: " + trainAccuracy);
            double testAccuracy = leNet5.testNetwork(testImages, testLabels);
            System.out.println("Test accuracy: " + testAccuracy);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static List<int[][]> readImages(String filePath) throws IOException {
        List<int[][]> allImages = new ArrayList<>();

        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(filePath))) {

            int magicNumber = readInt(bis);
            int numberOfImages = readInt(bis);
            int numberOfRows = readInt(bis);
            int numberOfColumns = readInt(bis);

            for (int i = 0; i < numberOfImages; i++) {
                int[][] image = new int[numberOfRows][numberOfColumns];
                for (int row = 0; row < numberOfRows; row++) {
                    for (int col = 0; col < numberOfColumns; col++) {
                        image[row][col] = bis.read() & 0xFF;
                    }
                }
                allImages.add(image);
            }

        }

        return allImages;
    }

    public static List<Integer> readLabels(String filePath) throws IOException {
        List<Integer> allLabels = new ArrayList<>();

        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(filePath))) {

            int magicNumber = readInt(bis);
            int numberOfItems = readInt(bis);

            for (int i = 0; i < numberOfItems; i++) {
                int label = bis.read() & 0xFF;
                allLabels.add(label);
            }

        }

        return allLabels;
    }

    private static int readInt(BufferedInputStream bis) throws IOException {
        return (bis.read() << 24) | (bis.read() << 16) | (bis.read() << 8) | bis.read();
    }

    public static void displayImage(int[][] image, String title) {
        JFrame window = new JFrame(title);
        int width = image[0].length;
        int height = image.length;
        window.setSize((width+2)*16, (height+2)*16 + 20);
        PixelGrid pGrid = drawImage(image, width, height);
        window.add(pGrid);
        window.setVisible(true);
        window.repaint();
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static PixelGrid drawImage(int[][] image, int width, int height) {
        PixelGrid grid = new PixelGrid(width, height);
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                grid.setPixel(image[i][j], j, i);
            }
        }
        return grid;
    }

    public static void displayImage(int[] image, String title) {
        JFrame window = new JFrame(title);
        int width = Math.max(image.length, 40);
        int height = 1;
        window.setSize((width+2)*16, (height+2)*16 + 20);
        PixelGrid pGrid = drawImage(image, width, height);
        window.add(pGrid);
        window.setVisible(true);
        window.repaint();
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static PixelGrid drawImage(int[] image, int width, int height) {
        PixelGrid grid = new PixelGrid(width, height);
        int c = 0;
        for(int i = 0; i<image.length; i++) {
            grid.setPixel(image[c++],1,i);
        }
        return grid;
    }
}
