package ML::CNN_pp;
use Modern::Perl;
# based on: https://zzutk.github.io/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf
use Math::Random qw(random_uniform);
use Data::Dumper;
use ML::Util qw(print_2d_array matmul transpose print_1d_array rotate_matrix_180 conv2d add_2_arrays);
use Cwd qw(abs_path);
use JSON;
use File::Slurp;

my $kernel_limit = 15;
my $c1_kernels_limit = 15;

sub new {
   my $class = shift;
   my $self = {};
   my %args = @_;
   if (defined($args{debug}) and $args{debug} == 1) {
      $self->{debug} = 1;
   } else {
      $self->{debug} = 0;
   }
   my $kernel_size = $args{kernel_size};
   if (!defined($kernel_size) or $kernel_size !~ /^\d+$/ or $kernel_size > $kernel_limit) {
      die "Invalid kernel_size parameter, must be integer <= $kernel_limit";
   }
   my $c1_kernels = $args{c1_kernels};  
   if (!defined($c1_kernels) or $c1_kernels !~ /^\d+$/ or $c1_kernels > $c1_kernels_limit) {
      die "Invalid c1_kernels parameter, must be integer <= $c1_kernels_limit";
   }
   my $input_dimensions = $args{input_dimensions};  
   if (!defined($input_dimensions) or ref($input_dimensions) ne "ARRAY" or $input_dimensions->[0] !~ /^\d+$/ or $input_dimensions->[1] !~ /^\d+$/) {
      die "Invalid input_dimensions parameter, must be array refs of 2 integers, e.g. [x, y]";
   }
   # so now we know the size and number of kernels in the first Convolution layer 
   my $kernels;
   my $c2_kernels = $c1_kernels * 2;
   my $c1_dist = sqrt($c1_kernels/((1 + $c1_kernels) * $kernel_size**2));
   my $initial_weights;
   if (defined($args{load_initial_weights}) and $args{load_initial_weights} == 1) {
      my $json_data = read_file("initial_weights.json");
      $initial_weights = from_json($json_data);
   }
   foreach my $p (0 .. $c1_kernels - 1) {
      foreach my $x (0 .. $kernel_size - 1) {
         foreach my $y (0 .. $kernel_size - 1) {
            $kernels->[0][$p][$x][$y] = random_uniform(1, -1 * $c1_dist , $c1_dist);
         }
      }
   }

# calculate sizes for various layers
   $self->{c1_kernels} = $c1_kernels;
   $self->{c2_kernels} = $c2_kernels; # are these known as "filters" in pytorch?
   $self->{kernel_size} = $kernel_size;
   $self->{range_upper} = ($self->{kernel_size} % 2 == 0)?($self->{kernel_size} / 2):( ($self->{kernel_size} - 1)/2 );
   $self->{range_lower} = -1 * $self->{range_upper};
   my $fc_input_size_side = $input_dimensions->[0]; 
   if (!$self->{padding}) {
      $fc_input_size_side -= $self->{range_upper} * 2; # after c1, the data will be kernel_size - 1 smaller each side
   }
   $self->{c1_size} = $fc_input_size_side; # needed for backprop
   $fc_input_size_side /= 2; # after first pooling size will be halved
   # this is the s1 size, we will need it later in backprop 
   $self->{s1_size} = $fc_input_size_side;
   if (!$self->{padding}) {
      $fc_input_size_side -= $self->{range_upper} * 2; # after c2, the data will be kernel_size - 1 smaller each side
   }
   # we will need c2_size on the backprop run
   $self->{c2_size} = $fc_input_size_side;
   $fc_input_size_side /= 2; # after second pooling size will be halved
   $self->{s2_size} = $fc_input_size_side;
   $self->{fc_input_size} = $fc_input_size_side * $fc_input_size_side * $self->{c2_kernels}; # output of second pooling is a square, and we have c2_kernels of them
   $self->{fc_output_size} = 10;

   my $biases;
   foreach my $p (0 .. $c1_kernels - 1) {
      $biases->[0][$p] = 0;
   }    
   foreach my $q (0 .. $c2_kernels - 1) {
      $biases->[1][$q] = 0;
   }    
   my $c2_dist = sqrt($c1_kernels/(($c1_kernels + $c2_kernels) * $kernel_size**2));
   foreach my $q (0 .. $c2_kernels - 1) {
      foreach my $p (0 .. $c1_kernels - 1) {
         foreach my $x (0 .. $kernel_size - 1) {
            foreach my $y (0 .. $kernel_size - 1) {
               $kernels->[1][$q][$p][$x][$y] = random_uniform(1, -1 * $c2_dist , $c2_dist);
            }
         }
      }
   }

   $self->{biases} = $biases;
   $self->{input_dimensions} = $input_dimensions;
   $self->{kernels} = $kernels;
   if (defined($args{load_initial_weights}) and $args{load_initial_weights} == 1) {
      $self->{kernels} = $initial_weights->{kernels};
   }
   $self->{c1} = []; # (input_dimension[0] - 2 * range_upper)^2, if no padding
   $self->{c2} = []; # (c1 - 2 * range_upper)^2, if no padding
   $self->{padding} = 0; # for future use, at the moment, no padding is applied, so C1 is kernel_size smaller than the input

# FC weights.  For demo purposes the output layer has 10 values.
   foreach my $b (0 .. $self->{fc_output_size} - 1) {
      $biases->[2][$b] = 0;
   }    

   my $w_dist = sqrt($c1_kernels/($self->{fc_input_size} + $self->{fc_output_size}));
   $self->{W} = [];
   if (defined($args{load_initial_weights}) and $args{load_initial_weights} == 1) {
      $self->{W} = $initial_weights->{weights};
   } else {
      foreach my $row ( 0 .. $self->{fc_output_size} - 1) {
         foreach my $col ( 0 .. $self->{fc_input_size} - 1) {
             $self->{W}[$row][$col] = random_uniform(1, -1 * $w_dist, $w_dist);
         }
      }
   }
   #print_2d_array("W", $self->{W}) if $self->{debug};
   return bless $self, $class;

}

sub load_label {
   my $self = shift;
   $self->{label} = shift;
}

sub forward {
   my $self = shift;
   my $input = shift;
   $self->{batch_size} = 1;
# convolution 1: input to c1 for p kernels
   $self->{input} = $input; # needed for backprop
   #print_2d_array("input", $self->{input}) if $self->{debug};

   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      $self->{c1}[$p] = conv2d($input, $self->{kernels}[0][$p], "reduce");
      print_2d_array("c1 kernel $p", $self->{kernels}[0][$p]) if $self->{debug};
      print_2d_array("c1 before activation $p", $self->{c1}[$p]) if $self->{debug};
      foreach my $i (0 .. $self->{c1_size} - 1) {
         foreach my $j (0 .. $self->{c1_size} - 1) {
            $self->{c1}[$p][$i][$j] = 1 / (1 + exp(-1 * ($self->{c1}[$p][$i][$j] + $self->{biases}[0][$p])));
         }
      }
      print_2d_array("c1 after activation $p", $self->{c1}[$p]) if $self->{debug};
   }

# pooling layer, p1: reduce c1 by 75%
   my $p1_upper_range = int(($self->{input_dimensions}[0] - 2 * $self->{range_upper}) / 2);
   say "pooling p1 size = $p1_upper_range" if $self->{debug};
   $self->{p1} = [];
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      print_2d_array("c1 $p", $self->{c1}[$p]) if $self->{debug};
      foreach my $i (0 .. $p1_upper_range - 1) {
         foreach my $j (0 .. $p1_upper_range - 1) {
            $self->{p1}[$p][$i][$j] = 0;
            foreach my $v (0 .. 1) {
               foreach my $u (0 .. 1) {
# algorithm defined as 1 relative, but perl arrays are 0 relative, so it is 2i + u, rather than 2i - u etc
                  $self->{p1}[$p][$i][$j] += $self->{c1}[$p][ 2 * $i + $u][2 * $j + $v]/4;
               }
            }
         }
      }
      print_2d_array("p1 $p", $self->{p1}[$p]) if $self->{debug};
   }
# convolution 2: p1 to c2 for q kernels

  foreach my $q (0 .. $self->{c2_kernels} - 1) {
     foreach my $i (0 .. $self->{c2_size} - 1) {
        foreach my $j (0 .. $self->{c2_size} - 1) {
           $self->{c2}[$q][$i][$j] = 0;
        }
     }
     foreach my $p (0 .. $self->{c1_kernels} - 1) {
        $self->{c2}[$q] = add_2_arrays($self->{c2}[$q], conv2d($self->{p1}[$p], $self->{kernels}[1][$q][$p], "reduce"));
     }
     print_2d_array("c2 $q before sigmoid", $self->{c2}[$q]) if $self->{debug};
     foreach my $i (0 .. $self->{c2_size} - 1) {
        foreach my $j (0 .. $self->{c2_size} - 1) {
           $self->{c2}[$q][$i][$j] = 1 / ( 1 + exp( -1 * ( $self->{c2}[$q][$i][$j] + $self->{biases}[1][$q] )));
        }
     }
     print_2d_array("c2 $q after sigmoid", $self->{c2}[$q]) if $self->{debug};
  } 

# 2nd pooling layer, p2: reduce c2 by 75%
   my $p2_upper_range = int(($p1_upper_range - 2 * $self->{range_upper}) / 2);
   $self->{p2_upper_range} = $p2_upper_range;
   say "pooling p2 size = $p2_upper_range" if $self->{debug};
   $self->{p2} = [];
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
         foreach my $i (0 .. $p2_upper_range - 1) {
            foreach my $j (0 .. $p2_upper_range - 1) {
               $self->{p2}[$q][$i][$j] = 0;
            }
         }
   }
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      print_2d_array("c2 $q", $self->{c2}[$q]) if $self->{debug};
#      foreach my $p (0 .. $self->{c1_kernels} - 1) {
         foreach my $i (0 .. $p2_upper_range - 1) {
            foreach my $j (0 .. $p2_upper_range - 1) {
#say "processing p2 [$q] cell [$i][$j]" if $self->{debug};
               foreach my $v (0 .. 1) {
                  foreach my $u (0 .. 1) {
# algorithm defined as 1 relative, but perl arrays are 0 relative, so it is 2i + u, rather than 2i - u etc
#say "adding " . $self->{c2}[$q][ 2 * $i + $u][2 * $j + $v] . "/4" if $self->{debug};
                     $self->{p2}[$q][$i][$j] += $self->{c2}[$q][ 2 * $i + $u][2 * $j + $v]/4;
                  }
               }
            }
         }
#      }
      print_2d_array("p2 $q", $self->{p2}[$q]) if $self->{debug};
   }
# vectorisation
   $self->{f} = [];
   my $row = 0;
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $j (0 .. $p2_upper_range - 1) { # column first
         foreach my $i (0 .. $p2_upper_range - 1) {
# each output must be 1 item per column, not 1 item per row.  So when we have a batch the matrix will be "batch entries" wide
            push @{$self->{f}[$row++]}, $self->{p2}[$q][$i][$j]; 
         }
      }
   }
# fully connected layer
   print_2d_array("f", $self->{f}) if $self->{debug};
   $self->{y} = matmul($self->{W}, $self->{f});  
   print_2d_array("y before sigmoid", $self->{y}) if $self->{debug};

   print_1d_array("fc biases", $self->{biases}[2]) if $self->{debug};
say "fc biases = " . Dumper($self->{biases}[2]);
foreach my $i (0 .. 9) {
   say "fc bias $i = " . $self->{biases}[2][$i];
}
   foreach my $row (0 .. $self->{fc_output_size} - 1) {
      foreach my $col (0 .. $self->{batch_size} - 1) {
say "idx = $row, value = " . $self->{y}->[$row][$col] . ", bias = " . $self->{biases}[2][$row];
         #$value = 1/(1 + exp( -1 * ($value + $self->{biases}[2][$bias_idx++])));
         $self->{y}->[$row][$col] = 1/(1 + exp( -1 * ($self->{y}->[$row][$col] + $self->{biases}[2][$row])));
say "updated value = " . $self->{y}->[$row][$col];
      }
   }
   print_2d_array("y after sigmoid", $self->{y}) if $self->{debug};
}
       
sub loss {
   my $self = shift;

   $self->{loss} = 0;
print_2d_array("y", $self->{y}) if $self->{debug};
print_2d_array("label", $self->{label}) if $self->{debug};
   foreach my $row (0 .. $#{$self->{y}}) {
say "row = $row, calc = " . $self->{y}[$row][0] . " , label = " . $self->{label}[$row][0] . ", difference = " . ($self->{y}[$row][0] - $self->{label}[$row][0]) if $self->{debug};
       $self->{loss} += 0.5 * ($self->{y}[$row][0] - $self->{label}[$row][0])**2;
   }
   return $self->{loss};
}

sub backward {
   my $self = shift;
   my $f_T = transpose($self->{f});
   print_2d_array("f_T", $f_T) if $self->{debug};
   $self->{delta_bias_fc} ||= [];
   foreach my $row(0 .. $#{$self->{y}}) {
      $self->{delta_bias_fc}[$row][0] = ($self->{y}[$row][0] - $self->{label}[$row][0]) * $self->{y}[$row][0] * (1 - $self->{y}[$row][0]);
   }

   print_2d_array("delta fc bias", $self->{delta_bias_fc}) if $self->{debug};
   $self->{delta_W} = matmul($self->{delta_bias_fc}, $f_T);

   print_2d_array("W", $self->{W}) if $self->{debug};
   print_2d_array("delta W", $self->{delta_W}) if $self->{debug};

   my $W_T = transpose($self->{W});
   $self->{delta_f} = matmul($W_T, $self->{delta_bias_fc});
   print_2d_array("W_T", $W_T) if $self->{debug};
   print_2d_array("delta_f", $self->{delta_f}) if $self->{debug};

# de-vectorisation, turn delta_f back into a series of pooled kernel deltas, i.e. the opposite of "vectorisation" above.
   $self->{delta_p2} ||= [];
   my $row = 0;
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $j (0 .. $self->{p2_upper_range} - 1) { 
         foreach my $i (0 .. $self->{p2_upper_range} - 1) { 
            $self->{delta_p2}[$q][$i][$j] =  $self->{delta_f}[$row++][0]; 
         }
      }
      print_2d_array("delta_p2 $q", $self->{delta_p2}[$q]) if $self->{debug};
   }
   $self->{delta_c2} ||= [];
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $i (0 .. $self->{c2_size} - 1) {
         foreach my $j (0 .. $self->{c2_size} - 1) {
            $self->{delta_c2}[$q][$i][$j] = 0.25 * $self->{delta_p2}[$q][int($i)/2][int($j/2)];
         }
      }
      print_2d_array("delta_c2 $q", $self->{delta_c2}[$q]) if $self->{debug};
      print_2d_array("c2 $q", $self->{c2}[$q]) if $self->{debug};
   }
   $self->{delta_c2_sigmoid} ||= [];
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $i (0 .. $self->{c2_size} - 1) {
         foreach my $j (0 .. $self->{c2_size} - 1) {
            $self->{delta_c2_sigmoid}[$q][$i][$j] = $self->{delta_c2}[$q][$i][$j] * ($self->{c2}[$q][$i][$j] * ( 1 - $self->{c2}[$q][$i][$j] ));
            #say "self->{delta_c2_sigmoid}[$q][$i][$j] ( " . $self->{delta_c2_sigmoid}[$q][$i][$j] . ") = self->{delta_c2}[$q][$i][$j] * (self->{c2}[$q][$i][$j] * ( 1 - self->{c2}[$q][$i][$j] )) = " . $self->{delta_c2}[$q][$i][$j] . " * (" . $self->{c2}[$q][$i][$j] . " * (1 - " . $self->{c2}[$q][$i][$j] . "))" if $self->{debug};
         }
      }
      print_2d_array("delta_c2_sigmoid $q", $self->{delta_c2_sigmoid}[$q]) if $self->{debug};
   }
   $self->{delta_k2} = [];
   my $p1_rot = []; # p1 rotated by 180 (known as S1 in the paper describing the algorithm)
   foreach my $p (0 .. $self->{c1_kernels}) {
      $p1_rot->[$p] = rotate_matrix_180($self->{p1}[$p]);
   }
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $p (0 .. $self->{c1_kernels} - 1) {
         print_2d_array("p1_rot $p", $p1_rot->[$p]) if $self->{debug};
         print_2d_array("delta_c2_sigmoid $q", $self->{delta_c2_sigmoid}[$q]) if $self->{debug};
         $self->{delta_k2}[$q][$p] = conv2d($p1_rot->[$p], $self->{delta_c2_sigmoid}[$q], "reduce");
         print_2d_array("delta_k2 $q/$p", $self->{delta_k2}[$q][$p]) if $self->{debug};
      }
   }

   $self->{delta_b2} = [];
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      $self->{delta_b2}[$q][0] = 0;
      foreach my $i (0 .. $self->{c2_size} - 1) {
         foreach my $j (0 .. $self->{c2_size} - 1) {
            $self->{delta_b2}[$q][0] += $self->{delta_c2_sigmoid}[$q][$i][$j];
say "delta_b2 $q adding delta_c2_sigmoid $q $i $j " . $self->{delta_c2_sigmoid}[$q][$i][$j] . " = " . $self->{delta_b2}[$q][0] if $self->{debug};
         }
      }
   }

   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $p (0 .. $self->{c1_kernels} - 1) {
         print_2d_array("delta_c2_sigmoid $q", $self->{delta_c2_sigmoid}[$q]) if $self->{debug};
         print_2d_array("p1 $p", $self->{p1}[$p]) if $self->{debug};
         print_2d_array("p1_rot $p", $p1_rot->[$p]) if $self->{debug};
         print_2d_array("delta_k2 $q/$p", $self->{delta_k2}[$q][$p]) if $self->{debug};
      }
      say "delta_b2 $q = " . $self->{delta_b2}[$q][0] if $self->{debug};
   }
   my $k2_rot = [];
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $p (0 .. $self->{c1_kernels} - 1) {
         $k2_rot->[$q][$p] = rotate_matrix_180($self->{kernels}[1][$q][$p]);
      }
   }
   
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      foreach my $q (0 .. $self->{c2_kernels} - 1) {
         print_2d_array("k2_rot $q/$p", $k2_rot->[$q][$p]) if $self->{debug};
         say "\n\n" if $self->{debug};
      }
   }
   $self->{p1_delta} = [];
   foreach my $u (0 .. $self->{s1_size} - 1) {
      foreach my $v (0 .. $self->{s1_size} - 1) {
         foreach my $p (0 .. $self->{c1_kernels}) {
             $self->{p1_delta}[$p][$u][$v] = 0;
         }
      }
   }
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      foreach my $q (0 .. $self->{c2_kernels} - 1) {
         $self->{p1_delta}[$p] = add_2_arrays($self->{p1_delta}[$p], conv2d($self->{delta_c2_sigmoid}[$q], $k2_rot->[$q][$p], "expand"));
      }
   }
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      print_2d_array("p1_delta $p",$self->{p1_delta}[$p]) if $self->{debug};
   }
   
   $self->{delta_c1} ||= [];

   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      foreach my $i (0 .. $self->{c1_size} - 1) {
         foreach my $j (0 .. $self->{c1_size} - 1) {
            $self->{delta_c1}[$p][$i][$j] = 0;
         }
      }
   }

   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      foreach my $i (0 .. $self->{c1_size} - 1) {
         foreach my $j (0 .. $self->{c1_size} - 1) {
            $self->{delta_c1}[$p][$i][$j] += 0.25 * $self->{p1_delta}[$p][int($i)/2][int($j/2)];
         }
      }
      print_2d_array("delta_c1 $p", $self->{delta_c1}[$p]) if $self->{debug};
   }

   $self->{delta_c1_sigmoid} ||= [];
   
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      foreach my $i (0 .. $self->{c1_size} - 1) {
         foreach my $j (0 .. $self->{c1_size} - 1) {
#            say "delta_c1_sigmoid->[$p][$i][$j] = delta_c1->[$p][$i][$j] * self->{c1}[$p][$i][$j] * ( 1 - self->{c1}[$p][$i][$j] )";
            $self->{delta_c1_sigmoid}[$p][$i][$j] = $self->{delta_c1}[$p][$i][$j] * $self->{c1}[$p][$i][$j] * ( 1 - $self->{c1}[$p][$i][$j] );
         }
      }
      print_2d_array("c1_delta_sigmoid $p", $self->{delta_c1_sigmoid}[$p]) if $self->{debug};
   }
   my $I_rot = rotate_matrix_180($self->{input});
   $self->{delta_k1} = [];
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      foreach my $u (0 .. $self->{kernel_size} - 1) {
         foreach my $v (0 .. $self->{kernel_size} - 1) {
            $self->{delta_k1}[$p][$u][$v] = 0;
         }
      }
   }
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      $self->{delta_k1}[$p] = conv2d($I_rot, $self->{delta_c1_sigmoid}[$p], "reduce");
      print_2d_array("delta_k1 $p", $self->{delta_k1}[$p]) if $self->{debug};
   }
   
   $self->{delta_b1} ||= [];
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      $self->{delta_b1}[$p][0] = 0;
      foreach my $i (0 .. $self->{c1_size} - 1) {
         foreach my $j (0 .. $self->{c1_size} - 1) {
            $self->{delta_b1}[$p][0] += $self->{delta_c1_sigmoid}[$p][$i][$j];
         }
      }
   }
   print_2d_array("b1", $self->{delta_b1}) if $self->{debug};

}

sub update_weights_and_biases {
   my $self = shift;
   my %params = @_;
   $params{batch_size} ||= 10;
   $params{learning_rate} ||= 3;
   $params{decay} ||= 1;
# update k1/p kernels
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      print_2d_array("kernels 0 $p before update", $self->{kernels}[0][$p]) if $self->{debug};
      print_2d_array("delta k1 $p", $self->{delta_k1}[$p]) if $self->{debug};
      foreach my $x (0 .. $self->{kernel_size} - 1) {
         foreach my $y (0 .. $self->{kernel_size} - 1) {
            $self->{kernels}[0][$p][$x][$y] -= $params{learning_rate} * $self->{delta_k1}[$p][$x][$y];
         }
      }
      print_2d_array("kernels 0 $p after update", $self->{kernels}[0][$p]) if $self->{debug};
   }
# update b1
   print_1d_array("biases 0 before update", $self->{biases}[0]) if $self->{debug};
   print_2d_array("delta b1", $self->{delta_b1}) if $self->{debug};
   foreach my $p (0 .. $self->{c1_kernels} - 1) {
      $self->{biases}->[0][$p] -= $params{learning_rate} * $self->{delta_b1}[$p][0];
   }
   print_1d_array("biases 0 after update", $self->{biases}[0]) if $self->{debug};
# update k2 p/q
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      foreach my $p (0 .. $self->{c1_kernels} - 1) {
         print_2d_array("kernels 1 $q $p before update", $self->{kernels}[1][$q][$p]) if $self->{debug};
         print_2d_array("delta_k2 $q $p", $self->{delta_k2}[$q][$p]) if $self->{debug};
         foreach my $x (0 .. $self->{kernel_size} - 1) {
            foreach my $y (0 .. $self->{kernel_size} - 1) {
               $self->{kernels}[1][$q][$p][$x][$y] -= $params{learning_rate} * $self->{delta_k2}[$q][$p][$x][$y];
            }
         }
         print_2d_array("kernels 1 $q $p after update", $self->{kernels}[1][$q][$p]) if $self->{debug};
      }
   }
#update b2
   print_1d_array("biases 1 before update", $self->{biases}[1]) if $self->{debug};
   print_2d_array("delta_b2", $self->{delta_b2}) if $self->{debug};
   foreach my $q (0 .. $self->{c2_kernels} - 1) {
      $self->{biases}[1][$q] -= $params{learning_rate} * $self->{delta_b2}[$q][0];
   }
   print_1d_array("biases 1 after update", $self->{biases}[1]) if $self->{debug};
#update W
   print_2d_array("W before update", $self->{W}) if $self->{debug};
   print_2d_array("W delta", $self->{delta_W}) if $self->{debug};
   foreach my $row ( 0 .. $self->{fc_output_size} - 1) {
      foreach my $col ( 0 .. $self->{fc_input_size} - 1) {
          $self->{W}[$row][$col] -= $params{learning_rate} * $self->{delta_W}[$row][$col];
      }
   }
   print_2d_array("W after update", $self->{W}) if $self->{debug};
# update output bias
   print_1d_array("biases fc before update", $self->{biases}[2]) if $self->{debug};
   print_2d_array("delta_biases_fc", $self->{delta_bias_fc}) if $self->{debug};
   foreach my $b (0 .. $self->{fc_output_size} - 1) {
      $self->{biases}[2][$b] -= $params{learning_rate} * $self->{delta_bias_fc}[$b][0];
   }    
   print_1d_array("biases fc after update", $self->{biases}[2]) if $self->{debug};
}
1;
