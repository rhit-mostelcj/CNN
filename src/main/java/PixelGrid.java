
import java.awt.image.BufferedImage;
import java.awt.Graphics;
import javax.swing.JPanel;

public class PixelGrid extends JPanel {

    public BufferedImage grid;
    int PIXEL_SIZE = 16;

    public PixelGrid(int width, int height) {
        grid = new BufferedImage(width*PIXEL_SIZE, height*PIXEL_SIZE, BufferedImage.TYPE_BYTE_GRAY);
    }

    public void setPixel(int color, int x, int y) {
//        color = 255-color;
        color = color == 0 ? 255 : 0;
        x = convertGridIndexToActual(x);
        y = convertGridIndexToActual(y);
        for(int i = 0; i<PIXEL_SIZE; i++) {
            for(int j = 0; j<PIXEL_SIZE; j++) {
                grid.setRGB((x + i)%grid.getWidth(), (y+ j) % grid.getHeight() , (int)(color << 16 | color << 8 | color));
            }
        }
    }

    @Override
    public void paintComponent(Graphics g){
        super.paintComponent(g);
        g.drawImage(grid, 10, 10, this);
    }

    public  int convertGridIndexToActual(int x) {
        return x*PIXEL_SIZE;
    }
}