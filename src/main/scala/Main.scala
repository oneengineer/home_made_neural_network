import org.nd4j.linalg.api.rng.DefaultRandom
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * Created by tedrahedron on 4/5/16.
  */
object Main extends App{


  val r = new DefaultRandom()
  val img2 = Nd4j.rand(28,28,0.0,1.0, r )


  //Util.plot_image( img2 )


  def read_digit() = {
    val f = scala.io.Source.fromFile("/Users/tedrahedron/work/handwriting/train_digit.csv")

    val v = for ( line <- f.getLines().take(10000).toVector) yield {
      val arr = line.split(" ")
      val label = arr.head.toInt
      val data = arr.tail.map(_.toDouble)

      val ndarr = (Nd4j.create(data).T - 0.0) / 256.0

      label -> ndarr
    }

    v.map(_._1) -> v.map(_._2)
  }


  def work1 = {
    val (label,data) = read_digit()

    val v1 = VariableLayer( data.head )
    val l1 = FullyConnected(v1,20)
    val l1s = Sigmoid(l1)

//    val l2 = FullyConnected(l1s,10)
//    val l2s = Sigmoid(l2)

    val o1 = FullyConnected(l1s,10)
    val o2 = SoftmaxLoss(o1)



    def forward() = {
      l1.forward()
      l1s.forward()
      //l2.forward()
      //l2s.forward()
      o1.forward()

      o2.forward()
    }

    def backward() = {
      o2.backward()
      o1.backward()
      //l2s.backward()
      //l2.backward()
      l1s.backward()
    }

    def updateweight() = {
      l1.updateweight()
      l1s.updateweight()
      //l2.updateweight()
      //l2s.updateweight()
      o1.updateweight()
      o2.updateweight()
    }

    val epoch = 20

    val ln = 400

    println(s"Data size ${data.size}")

    val train = data take ln
    val train_label = label take ln

    val test = data drop ln
    val test_label = label drop ln

    var sum = 0

    for (turn <- 0 until epoch){
      println(s" Epoch $turn ")

      sum = 0

      for ( i <- 0 until train.size ){

        //println(s"Lable ${train_label(i)}")

        v1.output = train(i)
        o2.label = train_label(i)

        forward()

//        println("l1.output",l1.output)
//        println("l1s.output",l1s.output)


        backward()


//        println( o1.recedived_gradient )
//        println( l1s.recedived_gradient )
//        println( l1.recedived_gradient )

        updateweight()

        if ( o2.predict != o2.label ){
          sum += 1
        }

      }
      println("predict error #:",sum)
      sum = 0
      for ( i <- 0 until test.size ){

        //println(s"Lable ${train_label(i)}")

        v1.output = test(i)
        o2.label = test_label(i)
        forward()
        if ( o2.predict != o2.label ){
          sum += 1
        }
      }
      println("test error #:",sum)

      //println( l1.weight )
      //println( l2.weight )


    }



  }

  //work1


  def work2 = {
    val (label,data0) = read_digit()
    val data = data0 map { _.reshape(1,28,28) }

    val v1 = ConvVariable(1,(28,28))
    val c1 = Conv( v1,(3,3),(2,2),5 )
    val c1s = ConvSigmoid(c1)

    val c2 = Conv( c1s,(3,3),(2,2),5 )
    val c2s = ConvSigmoid(c2)

    val fl = Flatten( c2s )
    val full1 = FullyConnected(fl,10)
    val o = SoftmaxLoss(full1)

    val epoch = 10

    val ln = 500

    println(s"Data size ${data.size}")

    val train = data take ln
    val train_label = label take ln

    //val test = data drop ln
    //val test_label = label drop ln

    var sum = 0

    for (turn <- 0 until epoch){
      println(s"Epoch $turn  =====> ")
      sum = 0
      for ( i <- 0 until train.size){
        val t1 = System.currentTimeMillis()
        v1.output = train(i)
        o.label = train_label(i)
        //println(s"Label ${o.label}")
        c1.forward()

        c1s.forward()

        c2.forward()
        c2s.forward()
        fl.forward()
        full1.forward()
        o.forward()

        //println("Output",o.output)

        o.backward()
        full1.backward()
        fl.backward()
        c2s.backward()
        c2.backward()
        c1s.backward()

        //println(c1.kernel_weights)
        //println("\n",c1.kernel_weights)

        c1.updateweight()
        c1s.updateweight()
        fl.updateweight()
        full1.updateweight()
        c2s.updateweight()
        c2.updateweight()

        //Util.debug(c2s.output)

        //println( "Weight" )
        //Util.debug( c1.kernel_weights.head )

        //Util.debug( c2.kernel_weights.head )


        if (o.predict != o.label){
          sum += 1
        }

        //println(" T cost", System.currentTimeMillis() - t1)

      }
      full1.step_size *= 0.8
      c2.step_size *= 0.9


      println(s"Error #: $sum")

      c1.kernel_weights foreach println

      Util.plot_images( c1.kernel_weights )

    }

  }

  work2

}
