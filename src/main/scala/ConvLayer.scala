import org.nd4j.linalg.factory.Nd4j

/**
  * Created by tedrahedron on 4/7/16.
  */
trait ConvLayer {

                // channel, y, x
  val img_shape:(Int,Int,Int)
  def channel:Int

}

import org.nd4j.linalg.convolution.DefaultConvolutionInstance

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import Util._


case class ConvVariable(val channel:Int,shape:(Int,Int)) extends Layer with ConvLayer{
  override val output_size: Int = -1

  override val img_shape: (Int, Int, Int) = ( channel,shape._1,shape._2 )

}

case class Conv(
                 layer: Layer with ConvLayer,
                 kernel:(Int,Int),
                 stride:(Int,Int),
                 override val channel:Int,
                 padding:(Int,Int) = (0,0)
               ) extends Layer with ConvLayer{
  override val output_size: Int = -1

  val kernel_weights = 1 to channel map {_ => Nd4j.rand( Array( layer.channel ,kernel._1,kernel._2),Config.r ) * 0.5 }

  val kernel_weights_b = Nd4j.rand( channel,1,10L) *  0.0


  override val img_shape = (channel ,
                        ( layer.img_shape._2 - kernel._1 ) / stride._1 + 1,
                        ( layer.img_shape._3 - kernel._2 ) / stride._2 + 1)

  this.recedived_gradient = Nd4j.zeros( tuple_to_array(img_shape) :_* )

  output = Nd4j.zeros( tuple_to_array(img_shape) :_* )

  override def forward(): Unit = {

    //parallel channel
    val par_channel = (0 until channel).toVector.par

    for ( c <- par_channel ) {
      var y2 = 0
      var x2 = 0
      for ( y <- 0 until img_shape._2 ){
        x2 = 0
        for (x <- 0 until img_shape._3 ){

          //debug
          //println(s"kernel_weights($c)")
          //debug(kernel_weights(c))

          val value = kernel_weights(c) * layer.output( ->,y2 -> (y2+kernel._1), x2 -> (x2+kernel._2) )
          output(c,y,x) = value.sumNumber().doubleValue()

          x2 += stride._2
        }
        y2 += stride._1
      }
    }
  }

  override def backward(): Unit = {
    var y2 = 0
    var x2 = 0

    // parallel x,y

    for ( c <- 0 until channel ) {
      y2 = 0
      for ( y <- 0 until img_shape._2 ){
        x2 = 0
        for (x <- 0 until img_shape._3 ){
          layer.recedived_gradient( ->,y2 -> (y2+kernel._1), x2 -> (x2+kernel._2) ) +=
            kernel_weights(c) * recedived_gradient( c,y,x )
          // a tensor multiple a scalar
          // dy/dx = w

          x2 += stride._2
        }
        y2 += stride._1
      }
    }
  }

  override def updateweight(): Unit = {

    kernel_weights_b -= recedived_gradient.sumNumber().doubleValue() * step_size


    //parallel channel
    val par_channel = (0 until channel).toVector.par

    for ( c <- par_channel ) {
      var y2 = 0
      var x2 = 0
      for ( y <- 0 until img_shape._2 ){
        x2 = 0
        for (x <- 0 until img_shape._3 ){

          kernel_weights(c) -=
               layer.output( ->,y2 -> (y2+kernel._1), x2 -> (x2+kernel._2) ) * recedived_gradient( c,y,x ) * step_size

          x2 += stride._2
        }
        y2 += stride._1
      }
    }

    recedived_gradient( -> ) = 0
  }



}


case class ConvSigmoid(
                        layer: Layer with ConvLayer
                      ) extends Layer with ConvLayer{
  override val output_size: Int = -1

  override def channel: Int = layer.channel

  override val img_shape: (Int, Int, Int) = layer.img_shape

  output = Nd4j.zeros( tuple_to_array(img_shape) :_* )

  this.recedived_gradient = Nd4j.zeros( tuple_to_array(img_shape) :_* )

  override def forward(): Unit = {
    output = layer.output.map{  x => 1.0/(  1.0 + math.exp( - x) ) }

  }

  override def backward(): Unit = {
    layer.recedived_gradient += (output map {x => x*(1-x) }) * recedived_gradient
  }

  override def updateweight(): Unit = {
    this.recedived_gradient( -> ) = 0
  }
}

case class Flatten( layer: Layer with ConvLayer ) extends Layer {

  override val output_size: Int = tuple_to_array(layer.img_shape).reduce( _ * _ )

  recedived_gradient = Nd4j.zeros(output_size).T

  override def forward(): Unit = {
    output = layer.output.reshape( output_size,1 )
  }

  override def backward(): Unit = {
    val temp = recedived_gradient.reshape( tuple_to_array(layer.img_shape):_* )
    layer.recedived_gradient += temp
  }

  override def updateweight(): Unit = {
    this.recedived_gradient( -> ) = 0
  }

}


