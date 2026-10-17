// Testbench for seq_detector (pattern 110101).
module seq_tb;

    reg  clk, rst, din;
    wire detected;

    integer out_file, t, c, seed;
    reg [2:0] state_before;

    parameter NUM_TESTS = 500;
    parameter SEQ_LEN   = 12;

    seq_dut DUT (.clk(clk), .rst(rst), .din(din), .detected(detected));

    always #5 clk = ~clk;

    initial begin
        clk  = 0;
        seed = 42;

        out_file = $fopen("results/seq_cov.csv", "w");
        if (out_file == 0) begin
            $display("ERROR: could not open file (does the results/ folder exist?)");
            $finish;
        end
        $fdisplay(out_file, "test_id,cycle,din,state_before,state_after,detected");

        for (t = 0; t < NUM_TESTS; t = t + 1) begin

            // reset the detector at the start of every test
            rst = 1; din = 0;
            @(negedge clk);
            rst = 0;

            // apply SEQ_LEN random bits, one per clock cycle
            for (c = 0; c < SEQ_LEN; c = c + 1) begin
                din          = $random(seed);
                state_before = DUT.state;
                @(posedge clk);
                #1;
                $fdisplay(out_file, "%0d,%0d,%0d,%0d,%0d,%0d",
                          t, c, din, state_before, DUT.state, detected);
                @(negedge clk);
            end
        end

        $fclose(out_file);
        $display("Done: %0d tests written to results/seq_cov.csv", NUM_TESTS);
        $finish;
    end

endmodule
