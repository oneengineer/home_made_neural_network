/**
  * Created by tedrahedron on 4/5/16.
  */

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import Util._

import org.nd4j.linalg.api.rng.distribution.impl._

trait Layer {

  def forward():Unit = {}
  def backward():Unit = {}
  def updateweight():Unit = {}

  var output:INDArray = null



  val output_size:Int


  var step_size:Double = 0.1

  var recedived_gradient:INDArray = Nd4j.zeros(output_size).T


}

object Config {

  val r = new NormalDistribution(0.0,1.0)
  r.reseedRandomGenerator(10L)
}


case class VariableLayer( val input:INDArray ) extends Layer {

  output = input
  override val output_size: Int = input.size(0)

  recedived_gradient = Nd4j.zeros(output_size).T

}


case class FullyConnected(layer:Layer,num_hidden:Int ) extends Layer {

  override val output_size: Int = num_hidden

  recedived_gradient = Nd4j.zeros(output_size).T

  val w_shape = Array( num_hidden,layer.output_size )

  private val init_dist = Config.r

  val weight:INDArray = Nd4j.rand( w_shape, init_dist ) - 0.5
  //val weight:INDArray = Nd4j.ones( w_shape:_* )

  val weight_b:INDArray = Nd4j.rand( Array(num_hidden,1), init_dist )


  val gradient:INDArray = Nd4j.zeros( output_size )

  override def forward() = {
    output = (weight ** layer.output) + weight_b
  }

  override def backward() = {

    val temp = weight.T ** recedived_gradient
    layer.recedived_gradient += temp
  }

  override def updateweight() = {

    weight_b -= recedived_gradient * step_size
    weight -= (recedived_gradient ** layer.output.T ) * step_size

    this.recedived_gradient( -> ) = 0.0
  }

}

case class Sigmoid(layer: Layer ) extends Layer{

  override val output_size: Int = layer.output_size

  recedived_gradient = Nd4j.zeros(output_size).T

  override def forward() = {
    output = layer.output.map{  x => 1.0/(  1.0 + math.exp( - x) ) }
  }

  override def backward() = {
    //println("sigmoid back ",(output map {x => x*(1-x) }) )

    layer.recedived_gradient += (output map {x => x*(1-x) }) * recedived_gradient
  }

  override def updateweight() = {
    this.recedived_gradient( -> ) = 0.0
  }

}

case class SoftmaxLoss( layer: Layer ) extends Layer{


  var label:Int = -1

  override val output_size: Int = layer.output_size

  var ex:INDArray = null

  var sum:Double = 0.0

  def predict = (0 to 9).zip(Util.ndarray_to_array(ex)).maxBy(_._2)._1

  override def forward() = {
    ex = layer.output map ( math.exp )

    sum = ex.sumNumber().doubleValue()

    ex /= sum

    output = ex
  }

  override def backward() = {

    val gradient = Nd4j.zeros(output_size).T

    gradient(label) = 1.0
    //the derivative of softmax is ex * (gradient - ex)

    //the dE/dz is
    layer.recedived_gradient = ex - gradient

  }

  override def updateweight() = {
    this.recedived_gradient( -> ) = 0.0
  }

}






