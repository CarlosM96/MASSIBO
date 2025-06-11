`timescale 1ns/10ps
//////////////////////////////////////////////////////////////////////////////////
// University: UNIMIB 
// Engineer: Esteban Cristaldo, MSc
//
// Create Date: July 1, 2022, 5:51:46 PM
// Design Name: filtering_and_selftrigger
// Module Name: filtroIIR_V2
// Project Name: selftrigger@bicocca
// Target Devices: DAPHNE V1
//
//////////////////////////////////////////////////////////////////////////////////
module filtroIIR_V2(
	input wire clk,
	input wire reset, 
	input wire enable, 
	input wire signed[15:0] x,
    output wire signed[15:0] y
);
    
    parameter n = 21;
    
	reg signed [15:0] en_mux, out_reg, x_i;
	reg signed [64:0] mean;
	reg [64:0] counter;	
	(* dont_touch = "true" *) reg reset_reg;
	(* dont_touch = "true" *) reg enable_reg;
	
	wire finish_signal; 
	
	always @(posedge clk) begin
	   reset_reg <= reset; 
	   enable_reg <= enable;
	end 

	always @(posedge clk) begin 
		if(reset_reg) begin 
			mean <= 32'b0;
			counter <= 16'b0;
			out_reg <= 32'b0;
			x_i <= 16'b0;
		end else if(~finish_signal) begin     
		    x_i <= x; 
		    mean <= mean + x_i;
		    counter <= counter + 1'b1; 
		end else begin
		    out_reg <= (mean >> n); 
		    mean <= 32'b0;
		    counter <= 32'b0;
		end
	end

	always @(posedge clk) begin
		if(enable_reg) begin
			en_mux <= out_reg[15:0];
		end else begin 
			en_mux <= 16'b0;
		end
	end
	
	assign y = en_mux;
    assign finish_signal = (counter[n]);
endmodule
	

	  
