`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// University: UNIMIB 
// Engineer: Esteban Cristaldo, MSc
//
// Create Date: July 28, 2022, 9:11:03 AM
// Design Name: filtering_and_selftrigger
// Module Name: filtroIIR_movmean40_cfd.v
// Project Name: selftrigger@bicocca
// Target Devices: DAPHNE V1
//
//////////////////////////////////////////////////////////////////////////////////


module filtroIIR_movmean40_cfd(
    input wire clk,
	input wire reset,
	input wire n_1_reset,
    input wire enable,
    input wire signed[15:0] x,
    output wire trigger,
    output wire signed[15:0] y
    );

  parameter shift_delay = 15;
  parameter threshold = -99999;

  reg signed [17:0] n1, n2, n3, d1, d2;
  reg signed [24:0] x_1, x_2, y_1, y_2;
	reg signed[15:0] x_i, en_mux, resta;
	reg signed [2*16 - 1 : 0] y_delay_reg;
  reg signed [shift_delay*16 -1 : 0] y_shifted;
	reg trigger_threshold, trigger_crossover, trigger_reg;
	reg [7:0] counter_threshold;
	(* dont_touch = "true" *) reg reset_reg;
	(* dont_touch = "true" *) reg enable_reg;

	wire signed[24:0] w1, w4, w7, w12, w13, w15;
	wire signed [17:0] w2, w20, w8, w14, w16;
	wire signed[47:0] w3, w5, w9, w6, w10, w11, w17, w18, w19;
	wire signed [15:0] y_shifted_w;
	
	always @(posedge clk) begin
	   reset_reg <= reset; 
	   enable_reg <= enable;
	end 

	always @(posedge clk) begin
		if(reset_reg) begin
      
        // ********** FBK *********** //
		// n1 <= {3'b000,15'b001100011001011};
		// n2 <= {3'b111,15'b101000001110111}; // n2 0101111100010001
		// n3 <= {3'b000,15'b001100000101101}; // n3 001100000101101
		// d1 <= {3'b001,15'b111000110001001};
		// d2 <= {3'b111,15'b000110110111101}; // 111001001000010
		// ********** FBK ************ //

		// ********** HPK ************ //
		n1 <= {3'b000,15'b001111101000111}; 
		n2 <= {3'b111,15'b100010110111100}; // 0111010010000111
	    n3 <= {3'b000,15'b001111011000101}; 
		d1 <= {3'b001,15'b110100000010000}; 
		d2 <= {3'b111,15'b001011000000111 + 1'b1}; // 1101001111110000
		// ********** HPK ************ //
	  
	  x_i <= 16'b0;
      x_1 <= 25'b0;
      x_2 <= 25'b0;
			y_1 <= 25'b0;
			y_2 <= 25'b0;
		end else if (n_1_reset) begin
		  x_i <= 16'b0;
			x_1 <= 25'b0;
      x_2 <= 25'b0;
			y_1 <= 25'b0;
			y_2 <= 25'b0;
		end else if (enable_reg) begin
		  x_i <= x;
			x_1 <= w1;
			x_2 <= w4;
			y_1 <= w12;
			y_2 <= w13;
		end
	end

    always @(posedge clk) begin
        if (reset_reg) begin
            en_mux <= 16'b0;
            y_delay_reg <= 2*16'b0;
            y_shifted <= shift_delay*16'b0;
            trigger_reg <= 1'b0;
		end else if(enable_reg) begin
			en_mux <= w11[39:24] + $signed(4);
      //en_mux <= w11[40:25]; // + $signed(4);
      resta <= en_mux - y_shifted[(shift_delay*16-1) : (shift_delay*16-1) - 15 ];
      y_delay_reg <= {y_delay_reg [15 : 0], resta};
			y_shifted <= {y_shifted[(shift_delay*16-1) - 16 : 0], en_mux};
			trigger_reg <= (trigger_threshold && trigger_crossover);
		end else begin
			en_mux <= x;
		end
	end

	always @(posedge clk) begin
	    if (reset_reg || counter_threshold[7]) begin
			trigger_threshold <= 1'b0;
		end else if(enable_reg) begin
			if (($signed(en_mux) < $signed(threshold)) || trigger_threshold) begin
			     trigger_threshold <= 1'b1;
			end
			/*if(counter_threshold[7]) begin
			      trigger_threshold <= 1'b0;
			end*/
		end
	end

	always @(posedge clk) begin
	    if (reset_reg || counter_threshold[7]) begin
	        counter_threshold <= 8'b0;
		end else if(enable_reg && trigger_threshold) begin
			counter_threshold <= counter_threshold + 1'b1;
		end
	end

	always @(posedge clk) begin
	    if (reset_reg || counter_threshold[7]) begin
	        trigger_crossover <= 1'b0;
		end else if(enable_reg && trigger_threshold) begin
			if (($signed(y_delay_reg[15:0]) >= $signed(16'd0)) && ($signed(y_delay_reg[31:16]) < $signed(16'd0))) begin
			     trigger_crossover <= 1'b1;
			end
		end
	end

	assign w1 = {x_i,9'b0};
  //assign w1 = {1'b0,x_i[13:0],10'b0};
  assign w2 = n1;
  assign w3 = (w1*w2);
  assign w4 = x_1;
  assign w20 = n2;
  assign w5 = (w4*w20);
  assign w7 = x_2;
  assign w8 = n3;
  assign w9 = (w7*w8);
  assign w6 = w3 + w5;
  assign w10 = w6 + w9;
  assign w11 = w19 + w10;
  assign w12 = w11[39:15];
  //assign w12 = w11[40:15];
  assign w13 = y_1;
  assign w14 = d1;
  assign w15 = y_2;
  assign w16 = d2;
  assign w17 = (w15*w16);
  assign w18 = (w13*w14);
  assign w19 = w18 + w17;
  assign y = en_mux;
  assign y_shifted_w = y_shifted[(shift_delay*16-1) : (shift_delay*16-1) - 15 ];
  assign trigger = trigger_reg;
endmodule
