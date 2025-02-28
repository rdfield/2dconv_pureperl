use Modern::Perl;
use lib '.';
use ML::CNN_pp;
use ML::Util qw(print_2d_array print_1d_array);
use File::Slurp;
use List::Util qw/shuffle/;
use Data::Dumper;
use POSIX qw(strftime);

my $test_data_size = -1;
my $epochs = 30;
my $learning_rate = 0.1;
my $debug = 0;
my $save_epoch_weights = 0;
my $validation_batch = -1;
my $load_initial_weights = 0;

my $noshuffle = 1;

sub argmax {
   my $arr = shift;
   my @max;
   foreach my $i (0 .. $#{$arr->[0]}) {
      my $max = $arr->[0][$i];
      $max[$i] = 0;
      my $idx = 0;
      foreach my $j ( 0 .. $#$arr) {
         if ($arr->[$j][$i] > $max) {
            $max = $arr->[$j][$i];
            $max[$i] = $j; 
         }
      }
   }        
   return \@max;
}

use Data::Dumper;
package mnist_loader {
   use Data::Dumper;
   use lib '.';
   use ML::MNIST;
   sub load_data_wrapper {
       my $MNIST = ML::MNIST->new();
       my $train_data = $MNIST->load_train_data();
       my $train_labels = $MNIST->load_train_labels();
       my $test_data = $MNIST->load_test_data();
       my $test_labels = $MNIST->load_test_labels();
       my $training_data = [];
       my $validation_data = [];
       my $testing_data = [];
       foreach my $i (0 .. $#{$train_data}) {
          if ($i<50000) {
             push @$training_data, [ $train_data->[$i], $train_labels->[$i] ];
          } else {
             push @$validation_data, [ $train_data->[$i], $train_labels->[$i] ];
          }
       }
       foreach my $i (0 .. $#{$test_data}) {
          push @$testing_data, [ $test_data->[$i], $test_labels->[$i] ];
       }

       return ($training_data, $validation_data, $testing_data);
   }
};

$|++;

my ($training_data, $validation_data, $test_data) = mnist_loader::load_data_wrapper();


say "initialising CNN";
my $CNN = ML::CNN_pp->new(kernel_size => 5,
                       c1_kernels => 6,
                       input_dimensions => [ 28, 28 ],
                       debug => $debug, 
                       load_initial_weights => 1);
say "initialising CNN complete";
# reformat data item
#say Dumper($training_data->[0]);
my @images;
my @labels;
my @validation_data; 
if ($noshuffle == 1) { 
   @validation_data = @$test_data; 
} else { 
   @validation_data = shuffle(@$test_data);
}
if ($validation_batch == -1) {
   $validation_batch = scalar(@$validation_data);
}

my @v_images;
my @v_labels;
foreach my $item (0 .. $validation_batch - 1) {
   my $label;
   foreach my $y (0 .. $#{$validation_data->[$item][1]}) {
      $label->[$y][0] = $validation_data[$item][1][$y];
   } 
   push @v_labels, $label;
   my $image = [];
   foreach my $x (0 .. 27) {
      foreach my $y (0 .. 27) {
         $image->[$x][$y] = $validation_data[$item][0][$x * 28 + $y];
      }
   }
   push @v_images, $image;
}
if ($test_data_size == -1) {
   $test_data_size = scalar(@$training_data);
}
foreach my $e (0 .. $epochs - 1) { # epoch
   my @training_data;
   if ($noshuffle) {
      @training_data = @$training_data;
   } else {
      @training_data = shuffle(@$training_data);
   }
   foreach my $item (0 .. $test_data_size - 1) {
      my $label;
      foreach my $y (0 .. $#{$training_data[$item][1]}) {
         $label->[$y][0] = $training_data[$item][1][$y];
      } 
      push @labels, $label;
      my $image = [];
      foreach my $x (0 .. 27) {
         foreach my $y (0 .. 27) {
            $image->[$x][$y] = $training_data[$item][0][$x * 28 + $y];
         }
      }
      push @images, $image;
   }
# say Dumper($label);
# print_2d_array("image",$image);
# my $label = [ [0], [0], [0], [1], [0], [0], [0], [0], [0], [0] ];
# get an image, in 2D for Perl (C can be flat, and 2D emulated)
# my @data = split//,scalar(read_file("small_digit_3.dat"));
# die Dumper(\@data);
# foreach my $x (0 .. 27) {
#   foreach my $y (0 .. 27) {
#      $image->[$x][$y] = ord($data[$x * 28 + $y]);
#   }
# }
   foreach my $i (0 .. $test_data_size - 1) {
      $CNN->load_label($labels[$i]);
      $CNN->forward($images[$i]);
      say "loss = " . $CNN->loss();
      $CNN->backward();
      $CNN->update_weights_and_biases(batch_size => 1, learning_rate => $learning_rate);
      if ($i % 100 == 0) {
         print ".";
      }
   }
   print "\n";
   print strftime "%Y-%m-%d %H:%M:%S", localtime time;
   print "\n";
   if ($save_epoch_weights == 1) {
      open FILE, ">", "epoch_${e}_params.txt";
      print FILE "biases: " . Dumper($CNN->{biases});
      print FILE "kernels: " . Dumper($CNN->{kernels});
      print FILE "weights: " . Dumper($CNN->{W});
      close FILE;
   }
   say "loss = " .$CNN->{loss};
   my $successes = 0;
  
   foreach my $i (0 .. $validation_batch - 1) {
      $CNN->load_label($v_labels[$i]);
#print_2d_array("input $i", $v_images[$i]) if $debug == 1;
      $CNN->forward($v_images[$i]);
      #print_2d_array("label", $v_labels[$i]) if $debug == 1;
      print_2d_array("result", $CNN->{y}) if $debug == 1;
      my $max_calc_idx = argmax($CNN->{y});
      my $max_target_idx = argmax($v_labels[$i]);
say "success = calc = " . $max_calc_idx->[0] . " == target = " . $max_target_idx->[0];
      $successes++ if $max_calc_idx->[0] == $max_target_idx->[0];
   }
say "epoch $e successes = $successes: " . int( ($successes * 100 ) / $validation_batch ) . "%";
   #foreach my $i (0 .. $#$max_calc_idx) {
      #$successes++ if  $max_calc_idx->[$i] == $max_target_idx->[$i];
   #}
}
