import * as React from 'react';
import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  StatusBar,
  Image,
  TouchableOpacity,
  Button
} from 'react-native'
import * as tf from '@tensorflow/tfjs'
import { fetch } from '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as jpeg from 'jpeg-js'
import * as ImagePicker from 'expo-image-picker'
import Constants from 'expo-constants'
import * as Permissions from 'expo-permissions'

// import exampleImage from './assets/stand.jpg'
import exampleImage from './assets/flex.jpg'
// import exampleImage from './assets/squat.jpg'

import * as poseNet from '@tensorflow-models/pose-detection'
import * as posenet from '@tensorflow-models/posenet';
// import { ModelView } from "./src/ModelView";

import { Video, AVPlaybackStatus } from 'expo-av';
import video from './assets/test.mp4';
import video1 from './assets/2.mp4';
// import { LogLevel, RNFFmpeg, RNFFprobe } from 'react-native-ffmpeg';
import * as svgComponents from 'react-native-svg';
// import Svg, {Circle, Line} from 'react-native-svg';



let Posemodel;
let modelType;
const inputTensorWidth = 152;
const inputTensorHeight = 200;
let squatCount = 0;

class App extends React.Component {

  constructor(props) {
    super(props);
    this.videoRef = React.createRef();
  }

  state = {
    isTfReady: false,
    isModelReady: false,
    poseNetModel: false,

    predictions: null,
    image: null,
    status:({}),
    

  }

  async loadPoseNetModel() {
    Posemodel = poseNet.SupportedModels.MoveNet;
    modelType = poseNet.movenet.modelType.SINGLEPOSE_LIGHTNING;
    return await poseNet.createDetector(Posemodel, { modelType: modelType });
  }

  async componentDidMount() {
    await tf.ready()
    this.setState({
      isTfReady: true
    })
    // this.model = await mobilenet.load()
    this.setState({ isModelReady: true })

    const [poseNetModel] = await Promise.all([this.loadPoseNetModel()]);
    console.log("Created Detector");
    this.setState({
      poseNetModel
    });

    // console.log("ffmpeg loading")
    // const framerate = await RNFFmpeg.execute(`ffmpeg -i ${video}`);
    // console.log("framerate ", framerate);
    // const v = await RNFFmpeg.execute(
    //   `-i ${video} -vf fps=25 out%03d.jpg`
    // )
    // console.log('exited', v)

    this.getPermissionAsync()
  }

  getPermissionAsync = async () => {
    if (Constants.platform.ios) {
      const { status } = await Permissions.askAsync(Permissions.CAMERA_ROLL)
      if (status !== 'granted') {
        alert('Sorry, we need camera roll permissions to make this work!')
      }
    }
  }


  imageToTensor(rawImageData) {
    const TO_UINT8ARRAY = true
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY)
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3)
    let offset = 0 // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset]
      buffer[i + 1] = data[offset + 1]
      buffer[i + 2] = data[offset + 2]

      offset += 4
    }

    console.log("size of image")
    console.log(width, height)
    this.setState({width})
    this.setState({height})
    return tf.tensor3d(buffer, [height, width, 3])
  }

  classifyImage = async () => {
    try {
      // const imageAssetPath = Image.resolveAssetSource(this.state.image)
      // const response = await fetch(imageAssetPath.uri, {}, { isBinary: true })

      console.log("1")
      const exampleImageUri = Image.resolveAssetSource(exampleImage).uri
      const response = await fetch(exampleImageUri, {}, { isBinary: true })
      const rawImageData = await response.arrayBuffer()
      const imageTensor = this.imageToTensor(rawImageData)
      
      // const predictions = await this.model.classify(imageTensor)
      let pose;
      const flipHorizontal = Platform.OS !== 'ios';
      console.log("video ref")

      const poses = await this.state.poseNetModel.estimatePoses(imageTensor, {
          maxPoses: 1, //When maxPoses = 1, a single pose is detected
          flipHorizontal: flipHorizontal
      });
      console.log("detect!")
      pose = poses[0];
      console.log(pose.keypoints)

      this.setState({pose});
      tf.dispose([imageTensor]);

      // this.setState({ predictions })
    } catch (error) {
      console.log(error)
    }
  }

  selectImage = async () => {
    try {
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3]
      })

      if (!response.cancelled) {
        const source = { uri: response.uri }
        this.setState({ image: source })
        this.classifyImage()
      }
    } catch (error) {
      console.log(error)
    }
  }

  renderPose() {
    const {width, height} = this.state
    const MIN_KEYPOINT_SCORE = 0.2;
    const {pose} = this.state;
    console.log("rendering pose")
    // console.log(pose)
    if (pose != null) {
      console.log("rendered!")
        // console.log(this.state.isFirstTime)
        // console.log(pose)
        const keypoints = pose?.keypoints
            .filter(k => k?.score > MIN_KEYPOINT_SCORE)
            .map((k, i) => {
                // console.log("positions")
                // console.log(k);
                // console.log(k.x);
                // console.log(k.y);

                return <svgComponents.Circle
                    key={`skeletonkp_${i}`}
                    cx={k.x}
                    cy={k.y}
                    r='5'
                    strokeWidth='0'
                    fill='blue'
                />;

                // <Svg height="50%" width="50%" viewBox={`0 0 ${width} ${height}`} >
                //   <Circle 
                //     key={`skeletonkp_${i}`}
                //     cx={k.x}
                //     cy={k.y}
                //     r='2'
                //     strokeWidth='0'
                //     fill='blue'
                //   />
                // </Svg>
            });


        const adjacentKeyPoints = poseNet.util.getAdjacentPairs(Posemodel);
        const skeleton = adjacentKeyPoints?.map(([i, j]) => {
            const kp1 = pose.keypoints[i];
            const kp2 = pose.keypoints[j]; // If score is null, just show the keypoint.
            const score1 = kp1?.score != null ? kp1?.score : 1;
            const score2 = kp2?.score != null ? kp2?.score : 1;
            const scoreThreshold = MIN_KEYPOINT_SCORE || 0;

            if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
                // console.log("kp1")
                // console.log(kp1)
                // console.log("kp2")
                // console.log(kp2)
                return <svgComponents.Line
                    key={`skeletonls_${i}${j}`}
                    x1={kp1.x}
                    y1={kp1.y}
                    x2={kp2.x}
                    y2={kp2.y}
                    stroke='white'
                    strokeWidth='5'
                />;
            }
        });


        return <svgComponents.Svg height='100%' width='100%' scaleX={1} scaleY={1}
                                  viewBox={`0 0 ${width} ${height}`}>
            {skeleton}
            {keypoints}
            {/* <svgComponents.Text
                stroke="white"
                fill="white"
                fontSize="30"
                fontWeight="bold"
                x="80"
                y="30"
                textAnchor="middle"
            >
                This is the place to draw pose!
            </svgComponents.Text> */}


        </svgComponents.Svg>;
    } else {
        return null;
    }
  }

  renderPrediction = prediction => {
    return (
      <Text key={prediction.className} style={styles.text}>
        {prediction.className}
      </Text>
    )
  }

  render() {
    
    const { isTfReady, isModelReady, poseNetModel, predictions, image, status } = this.state

    return (
      
        
      <View style={styles.container}>
        <StatusBar barStyle='light-content' />
        

        {/* <View style={styles.container}>
          <Video
            ref = {this.videoRef}
            source={video}
            resizeMode = "cover"
            style = {StyleSheet.absoluteFill}
            useNativeControls
            // paused = {false}
            // repeat={true}
            isLooping 
            onPlaybackStatusUpdate={status => this.setState({status})}
          />  
          <Button
                onPress={() => status.isPlaying ? this.videoRef.current.pauseAsync() : this.videoRef.current.playAsync()}  
                title={status.isPlaying ? 'Stop' : 'Play'}  
          />
        </View>  */}
        
        {/* <TouchableOpacity
          style={styles.imageWrapper}
          onPress={isModelReady ? this.selectImage : undefined}>
          {image && <Image source={image} style={styles.imageContainer} />}

          {isModelReady && !image && (
            <Text style={styles.transparentText}>Tap to choose image</Text>
          )}
        </TouchableOpacity> */}
        {/* <View style={styles.predictionWrapper}>
          {isModelReady && image && (
            <Text style={styles.text}>
              Predictions: {predictions ? '' : 'Predicting...'}
            </Text>
          )}
          {isModelReady &&
            predictions &&
            predictions.map(p => this.renderPrediction(p))}
        </View> */}

        <View style={styles.cameraInnerContainer}>
          <Image source={exampleImage} style={styles.camera} />
          <View style={styles.modelResults}>
            {this.state.pose ? this.renderPose() : null}
          </View>
        </View>
        
        <Button 
                onPress={() => {this.classifyImage()}}  
                title='Detect'
        />

        <View style={styles.loadingContainer}>
          <Text style={styles.text}>
            TFJS ready? {isTfReady ? <Text>✅</Text> : ''}
          </Text>
          <View style={styles.loadingModelContainer}>
            <Text style={styles.text}>PoseNetModel ready? </Text>
            {poseNetModel ? (
              <Text style={styles.text}>✅</Text>
            ) : (
              <ActivityIndicator size='small' />
            )}
          </View>
        </View>

        <View style={styles.footer}>
          <Text style={styles.poweredBy}>Powered by:</Text>
          <Image source={require('./assets/tfjs.jpg')} style={styles.tfLogo} />
        </View>
      </View>
    )
  }
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#171f24',
    alignItems: 'center'
  },
  loadingContainer: {
    // position: 'absolute',
    marginTop: 40,
    justifyContent: 'center'
  },
  text: {
    color: '#ffffff',
    fontSize: 16
  },
  loadingModelContainer: {
    flexDirection: 'row',
    marginTop: 10
  },
  imageWrapper: {
    width: 280,
    height: 280,
    padding: 10,
    borderColor: '#cf667f',
    borderWidth: 5,
    borderStyle: 'dashed',
    marginTop: 40,
    marginBottom: 10,
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center'
  },
  imageContainer: {
    width: 250,
    height: 250,
    position: 'absolute',
    top: 10,
    left: 10,
    bottom: 10,
    right: 10
  },
  predictionWrapper: {
    height: 100,
    width: '100%',
    flexDirection: 'column',
    alignItems: 'center'
  },
  transparentText: {
    color: '#ffffff',
    opacity: 0.7
  },
  footer: {
    marginTop: 40
  },
  poweredBy: {
    fontSize: 20,
    color: '#e69e34',
    marginBottom: 6
  },
  cameraInnerContainer: {
    // position: 'absolute',
    marginTop: 20,              ////////??????????????????????????????????????
    flex: 1,
    width: '100%',
    zIndex : -1
  },
  camera: {
      position: 'absolute',
      left: 160,
      top: 40,                ////////??????????????????????????????????????
      width: 300,
      height: 400,
      // zIndex: 1,
      borderWidth: 1,
      borderColor: 'red',
      // borderRadius: 0,
  },
  modelResults: {
    position: 'absolute',
    left: 160,
    top: 40,
    width: 300,
    height: 400,
    zIndex: 20000,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  },
  tfLogo: {
    width: 125,
    height: 70
  }
})

export default App