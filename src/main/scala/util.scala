import java.util

import breeze.linalg.DenseMatrix

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.DefaultRandom

import org.nd4s.Implicits._

import org.nd4j.linalg.factory.Nd4j



import breeze.plot._

import scala.util.Random


object Util{

  implicit def tuple_to_array(shape:(Int,Int)):Array[Int] = Array(shape._1,shape._2)
  implicit def tuple_to_array(shape:(Int,Int,Int)):Array[Int] = Array(shape._1,shape._2,shape._3)


  def debug(nd:INDArray) = {
    println( "Shape: ",nd.shape().toVector,nd ,"\n")
  }

  def ndarray_to_array( arr:INDArray ) = {

    val shapearr = arr.shape()
    val size = shapearr.reduce( _ * _ )

    val w = Array.fill[Double](size)(0.0)


    var i = 0


    for( v <- arr ) yield {
      w(i) = v ; i += 1
      v
    }
    w
  }

  def plot_image(img:INDArray) = {

    val a = ndarray_to_array(img)

    val m = DenseMatrix.create( img.shape()(0),img.shape()(1), a)

    val f2 = Figure()
    f2.subplot(0) += image( m )
    f2.refresh()

  }

  def plot_images(imgs:Seq[INDArray]) = {


    val figs = imgs map { img=>
      val a = ndarray_to_array(img)

      val m = DenseMatrix.create( img.shape()(1),img.shape()(2), a)
      m
    }


    val f2 = Figure()
    f2.rows = 5
    figs.zipWithIndex foreach { case (m,idx) =>
      f2.subplot(idx) += image( m )
    }

    f2.refresh()

  }

}
