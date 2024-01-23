package tool;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * これはJavaのHello Worldプログラムです。
 * ドックストリングはプログラムの説明やドキュメントを提供します。
 * このプログラムでは、forループを使用して3回Hello Worldを出力します。
 */
public class Test {
    public static void main(String[] args) {
        // forループを使用して3回Hello Worldを出力
        for (int i = 0; i < 3; i++) {
            System.out.println("Hello World!");
        }
    }
}
