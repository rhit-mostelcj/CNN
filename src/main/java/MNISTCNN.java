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
        try {
            List<int[][]> images = readImages(trainImagesPath);
            List<Integer> labels = readLabels(trainLabelsPath);

            LeNet5 leNet5 = new LeNet5(images, labels);
            int[][] testImage = images.get(0);
            int[][] paddedTestImage = leNet5.addPadding(testImage);

            print2DArray(testImage);
            System.out.println();
            print2DArray(paddedTestImage);

            displayImage(testImage);
            displayImage(paddedTestImage);

//            int c = 0;
//            for (int[][] image : images) {
//                if (c == 10) {
//                    break;
//                }
//                displayImage(image, image[0].length, image.length);
//                c++;
//            }
//
//            c = 0;
//            for (int label : labels) {
//                if (c == 10) {
//                    break;
//                }
//
//                System.out.println("Label " + (c + 1) + ": " + label);
//                c++;
//            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void print2DArray(int[][] array) {
        for (int row = 0; row < array.length; row++) {
            for (int col = 0; col < array[0].length; col++) {
                System.out.print(array[row][col] + " ");
            }
            System.out.println();
        }
    }

    public static List<int[][]> readImages(String filePath) throws IOException {
        List<int[][]> allImages = new ArrayList<>();

        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(filePath))) {

            // Read header
            int magicNumber = readInt(bis);
            int numberOfImages = readInt(bis);
            int numberOfRows = readInt(bis);
            int numberOfColumns = readInt(bis);

            System.out.println("Magic Number: " + magicNumber);
            System.out.println("Number of Images: " + numberOfImages);
            System.out.println("Image Size: " + numberOfRows + "x" + numberOfColumns);

            // Read images
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

            // Read header
            int magicNumber = readInt(bis);
            int numberOfItems = readInt(bis);

//            numberOfItems = TESTINGSIZE;

            System.out.println("Magic Number: " + magicNumber);
            System.out.println("Number of Labels: " + numberOfItems);

            // Read labels
            for (int i = 0; i < numberOfItems; i++) {
                int label = bis.read() & 0xFF; // Convert to unsigned byte
                allLabels.add(label);
            }

        }

        return allLabels;
    }

    private static int readInt(BufferedInputStream bis) throws IOException {
        return (bis.read() << 24) | (bis.read() << 16) | (bis.read() << 8) | bis.read();
    }

    public static void displayImage(int[][] image) {
        JFrame window = new JFrame("foo");
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
}
