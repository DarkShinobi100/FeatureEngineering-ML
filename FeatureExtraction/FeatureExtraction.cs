using System;
using System.Collections.Generic;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;

// CMP304: Artificial Intelligence  - Lab 2 Example Code

namespace FeatureExtraction
{
    // The main program class
    class Program
    {
        float DistanceAcrossFace = 0.0f;
        // file paths
        private const string inputFilePath = @"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\input.jpg";

        // The main program entry point
        static void Main(string[] args)
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

            System.Collections.Generic.List<float> FeatureVector = new List<float> { 0.0f };

            // Set up Dlib Face Detector
            using (var fd = Dlib.GetFrontalFaceDetector())
            // ... and Dlib Shape Detector
            using (var sp = ShapePredictor.Deserialize(@"C:\Users\User\Documents\C++\MachineLearning\FeatureEngineering\FeatureExtraction\shape_predictor_68_face_landmarks.dat"))
            {
                // load input image
                var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

                // find all faces in the image
                var faces = fd.Operator(img);
                // for each face draw over the facial landmarks
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // draw the landmark points on the image
                    for (var i = 0; i < shape.Parts; i++)
                    {
                        var point = shape.GetPart((uint)i);
                        var rect = new Rectangle(point);

                        //left eyebrow
                        if(i==19 || i==20 || i==21 ||i==22)
                        {
                            var Left = point - shape.GetPart((uint)40);
                            NonNormalisedLeftEyebrowDistance += (float)Left.Length;
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 0, 0), thickness: 4);
                        }
                        //Right eyebrow
                        else if (i == 23 || i == 24 || i == 25 || i == 26)
                        {
                            var Right = point - shape.GetPart((uint)43);
                            NonNormalisedRightEyebrowDistance += (float)Right.Length;
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(200, 0, 0), thickness: 4);
                        }
                        //left lip
                        else if (i == 49 || i == 50 || i == 51)
                        {
                            NonNormalisedLeftLipDistance += (float)(point - shape.GetPart((uint)34)).Length;
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 255, 0), thickness: 4);
                        }
                        //Right lip
                        else if (i == 53 || i == 54 || i == 55)
                        {
                            NonNormalisedRightLipDistance += (float)(point - shape.GetPart((uint)34)).Length;
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(0, 200, 0), thickness: 4);
                        }
                        else
                        {
                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                        }

                    }
                    LeftEyebrow = NonNormalisedLeftEyebrowDistance / (float)(shape.GetPart((uint)40) - shape.GetPart((uint)22)).Length;
                    RightEyebrow = NonNormalisedRightEyebrowDistance / (float)(shape.GetPart((uint)43) - shape.GetPart((uint)23)).Length;
                    LeftLip = NonNormalisedLeftLipDistance / (float)(shape.GetPart((uint)34) - shape.GetPart((uint)52)).Length;
                    RightLip = NonNormalisedRightLipDistance / (float)(shape.GetPart((uint)34) - shape.GetPart((uint)52)).Length;
                    LipWidth = (float)(shape.GetPart((uint)49) - shape.GetPart((uint)55)).Length / (float)(shape.GetPart((uint)34) - shape.GetPart((uint)52)).Length;
                    LipHeight = (float)(shape.GetPart((uint)52) - shape.GetPart((uint)58)).Length / (float)(shape.GetPart((uint)34) - shape.GetPart((uint)52)).Length;

                    //add values to vector
                    FeatureVector.Add(LeftEyebrow);
                    FeatureVector.Add(RightEyebrow);
                    FeatureVector.Add(LeftLip);
                    FeatureVector.Add(RightLip);
                    FeatureVector.Add(LipWidth);
                    FeatureVector.Add(LipHeight);

                    //print values for debug
                    Console.WriteLine("Left eyebrow:" + LeftEyebrow.ToString());
                    Console.WriteLine("Right eyebrow:" + RightEyebrow.ToString());
                    Console.WriteLine("Left lip:" + LeftLip.ToString());
                    Console.WriteLine("Right lip:" + RightLip.ToString());
                    Console.WriteLine("Lip width:" + LipWidth.ToString());
                    Console.WriteLine("Lip height:" + LipHeight.ToString());
                }
                // export the modified image
                Dlib.SaveJpeg(img, "output.jpg");

                //The header definiteion of the CSV file
                string header = "label, leftEyebrow,rightEyebrow,leftLip,rightLip,lipHeight,lipWidth\n";

                //create the CSV file and fill in the forst line with the header
                System.IO.File.WriteAllText(@"feature_vectors.csv", header);

                using(System.IO.StreamWriter file = new System.IO.StreamWriter(@"feature_vectors.csv",true))
                {
                    Console.WriteLine("Saving...");
                    file.WriteLine("" + "," + LeftEyebrow + "," + RightEyebrow + "," + LeftLip + "," + RightLip + "," + LipHeight + "," + LipWidth);
                }
            }
        }
    }
}