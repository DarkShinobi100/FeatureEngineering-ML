using System;
using System.Collections.Generic;
using System.Globalization;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using System.IO;

// CMP304: Artificial Intelligence  - Lab 2 Example Code

namespace FeatureExtraction
{
    // The main program class
    class Program
    {
        // file paths
        private const string inputFilePath = @"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\input.jpg";

        // The main program entry point
        static void Main(string[] args)
        {
            for (int i = 0; i < 7; i++)
            {

                if(i == 0)
                {
                    string[] files = Directory.GetFiles(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\Cohn-KanadeImages\000 neutral", "S*", SearchOption.AllDirectories);
                    TestData(files, "neutral", true);
                }
                else if (i == 1)
                {
                    string[] files = Directory.GetFiles(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\Cohn-KanadeImages\001 surprise", "S*", SearchOption.AllDirectories);
                    TestData(files, "surprise", false);
                }
                else if (i == 2)
                {
                    string[] files = Directory.GetFiles(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\Cohn-KanadeImages\002 sadness", "S*", SearchOption.AllDirectories);
                    TestData(files, "sadness", false);
                }
                else if (i == 3)
                {
                    string[] files = Directory.GetFiles(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\Cohn-KanadeImages\003 fear", "S*", SearchOption.AllDirectories);
                    TestData(files, "fear", false);
                }
                else if (i == 4)
                {
                    string[] files = Directory.GetFiles(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\Cohn-KanadeImages\004 anger", "S*", SearchOption.AllDirectories);
                    TestData(files, "anger", false);
                }
                else if (i == 5)
                {
                    string[] files = Directory.GetFiles(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\Cohn-KanadeImages\005 disgust", "S*", SearchOption.AllDirectories);
                    TestData(files, "disgust", false);
                }
                if (i == 6)
                {
                    string[] files = Directory.GetFiles(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\Cohn-KanadeImages\006 joy", "S*", SearchOption.AllDirectories);
                    TestData(files, "joy", false);
                }

            }
            Console.WriteLine("Done");
        }

        static void TestData(string[] Images, string Emotion, bool Start)
        {
            Console.WriteLine(Emotion);

            System.Collections.Generic.List<float> FeatureVector = new List<float> { 0.0f };
            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\shape_predictor_68_face_landmarks.dat"))
            {
                for (int i = 0; i < Images.Length; i++)
                {
                    //Variables
                    float LeftEyebrow = 0.0f;
                    float RightEyebrow = 0.0f;
                    float LeftLip = 0.0f;
                    float RightLip = 0.0f;
                    float LipWidth = 0.0f;
                    float LipHeight = 0.0f;

                    float NonNormalisedLeftEyebrowDistance = 0.0f;
                    float NonNormalisedRightEyebrowDistance = 0.0f;
                    float NonNormalisedLeftLipDistance = 0.0f;
                    float NonNormalisedRightLipDistance = 0.0f;

                    // load input image

                    var img = Dlib.LoadImage<RgbPixel>(Images[i]);
                    Console.WriteLine("Load Image:" + i.ToString());

                    // find all faces in the image
                    var faces = fd.Operator(img);
                    // for each face draw over the facial landmarks
                    foreach (var face in faces)
                    {
                        // find the landmark points for this face
                        var shape = sp.Detect(img, face);

                        // draw the landmark points on the image
                        for (var j = 1; j <= shape.Parts; j++)
                        {
                            var point = shape.GetPart((uint)j - 1);
                            var rect = new Rectangle(point);

                            //left eyebrow
                            if (j == 19 || j == 20 || j == 21 || j == 22)
                            {
                                var Left = point - shape.GetPart((uint)39);
                                NonNormalisedLeftEyebrowDistance += (float)Left.Length;
                                Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 0), thickness: 4);
                            }
                            //Right eyebrow
                            else if (j == 23 || j == 24 || j == 25 || j == 26)
                            {
                                var Right = point - shape.GetPart((uint)42);
                                NonNormalisedRightEyebrowDistance += (float)Right.Length;
                                Dlib.DrawRectangle(img, rect, color: new RgbPixel(200, 0, 0), thickness: 4);
                            }
                            //left lip
                            else if (j == 49 || j == 50 || j == 51)
                            {
                                NonNormalisedLeftLipDistance += (float)(point - shape.GetPart((uint)33)).Length;
                                Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 0), thickness: 4);
                            }
                            //Right lip
                            else if (j == 53 || j == 54 || j == 55)
                            {
                                NonNormalisedRightLipDistance += (float)(point - shape.GetPart((uint)33)).Length;
                                Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 200, 0), thickness: 4);
                            }
                            else
                            {
                                Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                            }

                        }
                        LeftEyebrow = NonNormalisedLeftEyebrowDistance / (float)(shape.GetPart((uint)21) - shape.GetPart((uint)39)).Length;
                        RightEyebrow = NonNormalisedRightEyebrowDistance / (float)(shape.GetPart((uint)22) - shape.GetPart((uint)42)).Length;
                        LeftLip = NonNormalisedLeftLipDistance / (float)(shape.GetPart((uint)51) - shape.GetPart((uint)33)).Length;
                        RightLip = NonNormalisedRightLipDistance / (float)(shape.GetPart((uint)51) - shape.GetPart((uint)33)).Length;
                        LipWidth = (float)(shape.GetPart((uint)54) - shape.GetPart((uint)48)).Length / (float)(shape.GetPart((uint)51) - shape.GetPart((uint)33)).Length;
                        LipHeight = (float)(shape.GetPart((uint)57) - shape.GetPart((uint)51)).Length / (float)(shape.GetPart((uint)51) - shape.GetPart((uint)33)).Length;

                        //add values to vector
                        FeatureVector.Add(LeftEyebrow);
                        FeatureVector.Add(RightEyebrow);
                        FeatureVector.Add(LeftLip);
                        FeatureVector.Add(RightLip);
                        FeatureVector.Add(LipWidth);
                        FeatureVector.Add(LipHeight);

                        //print values for debug
                        /*
                        Console.WriteLine("Left eyebrow:" + LeftEyebrow.ToString());
                        Console.WriteLine("Right eyebrow:" + RightEyebrow.ToString());
                        Console.WriteLine("Left lip:" + LeftLip.ToString());
                        Console.WriteLine("Right lip:" + RightLip.ToString());
                        Console.WriteLine("Lip width:" + LipWidth.ToString());
                        Console.WriteLine("Lip height:" + LipHeight.ToString());
                        Console.WriteLine("Files:" + Images.Length.ToString());
                        */
                    }
                    // export the modified image
                    //Dlib.SaveJpeg(img, "output.jpg");

                    //first run set up headers
                    if (i == 0 && Start)
                    {
                        //The header definiteion of the CSV file
                        string header = "label, leftEyebrow,rightEyebrow,leftLip,rightLip,lipHeight,lipWidth\n";

                        //create the CSV file and fill in the first line with the header
                        System.IO.File.WriteAllText(@"feature_vectors.csv", header);
                    }
                    using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"feature_vectors.csv", true))
                    {
                        Console.WriteLine("Saving...");
                        file.WriteLine(Emotion + "," + LeftEyebrow + "," + RightEyebrow + "," + LeftLip + "," + RightLip + "," + LipHeight + "," + LipWidth);
                        FeatureVector.Clear();
                    }
                }
            }
        }
    }
}