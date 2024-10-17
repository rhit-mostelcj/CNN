import javax.swing.*;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTCNN {

    private static final int TESTINGSIZE = 10;

    public static void main(String[] args) {
        String trainImagesPath = "train-images-idx3-ubyte";
        String trainLabelsPath = "train-labels-idx1-ubyte";
        try {
            readImages(trainImagesPath);
            readLabels(trainLabelsPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void readImages(String filePath) throws IOException {
        FileInputStream fis = new FileInputStream(filePath);

        // Read header
        int magicNumber = readInt(fis);
        int numberOfImages = readInt(fis);
        int numberOfRows = readInt(fis);
        int numberOfColumns = readInt(fis);

        numberOfImages = TESTINGSIZE;

        System.out.println("Magic Number: " + magicNumber);
        System.out.println("Number of Images: " + numberOfImages);
        System.out.println("Image Size: " + numberOfRows + "x" + numberOfColumns);

        // Read images
        for (int i = 0; i < numberOfImages; i++) {
            int[][] image = new int[numberOfRows][numberOfColumns];
            for (int row = 0; row < numberOfRows; row++) {
                for (int col = 0; col < numberOfColumns; col++) {
                    image[row][col] = fis.read() & 0xFF; // Convert to unsigned byte
                }
            }
            // Process the image here (e.g., display or save it)
            System.out.println("Read image " + (i + 1));
            displayImage(image, numberOfColumns, numberOfRows);
        }

        fis.close();
    }

    public static void readLabels(String filePath) throws IOException {
        FileInputStream fis = new FileInputStream(filePath);

        // Read header
        int magicNumber = readInt(fis);
        int numberOfItems = readInt(fis);

        numberOfItems = TESTINGSIZE;

        System.out.println("Magic Number: " + magicNumber);
        System.out.println("Number of Labels: " + numberOfItems);

        // Read labels
        for (int i = 0; i < numberOfItems; i++) {
            int label = fis.read() & 0xFF; // Convert to unsigned byte
            System.out.println("Label " + (i + 1) + ": " + label);
        }

        fis.close();
    }

    private static int readInt(FileInputStream fis) throws IOException {
        return (fis.read() << 24) | (fis.read() << 16) | (fis.read() << 8) | fis.read();
    }

    public static void displayImage(int[][] image, int width, int height) {
        JFrame window = new JFrame("foo");
        window.setSize((width+2)*16, (height+2)*16 + 20);
        PixelGrid pGrid = drawImage(image, width, height);
        window.add(pGrid);
        window.setVisible(true);
        window.repaint();
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static PixelGrid drawImage(int[][] image, int width, int height) {
        PixelGrid grid = new PixelGrid(width, height);
        for(int i = 0; i<28; i++) {
            for(int j = 0; j<28; j++) {
                grid.setPixel(image[i][j],j,i);
            }
        }
        return grid;
    }
}
