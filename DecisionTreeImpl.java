import java.util.List;
import java.lang.Double;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 */
public class DecisionTreeImpl {
	public DecTreeNode root;
	public List<List<Integer>> trainData;
	public int maxPerLeaf;
	public int maxDepth;
	public int numAttr;

	// Build a decision tree given a training set
	DecisionTreeImpl(List<List<Integer>> trainDataSet, int mPerLeaf, int mDepth) {
		this.trainData = trainDataSet;
		this.maxPerLeaf = mPerLeaf;
		this.maxDepth = mDepth;
		if (this.trainData.size() > 0) this.numAttr = trainDataSet.get(0).size() - 1;
		this.root = buildTree();
	}

	private int getLabel(List<Integer> record) {
		return record.get(record.size()-1);
	}

	private double log2(double a) {
		if(a == 0)
			return 0;
		return (Math.log(a)/Math.log(2));
	}

	private void computeEntropies(List<List<Integer>> currentData, List<Double> entropies, List<Integer> thresholds) {
		int attr;
		for(attr=0; attr<numAttr; attr++) {
			int threshold, bestThreshold=1;
			double bestEntropy = Double.POSITIVE_INFINITY;
			for(threshold=1; threshold<10; threshold++) {
				double p = 0, q = 0, u = 0, v = 0;
				for(List<Integer> record : currentData) {
					if(getLabel(record) == 0) {
						if(record.get(attr) <= threshold)
							p++;
						else
							q++;
					} else {
						if(record.get(attr) <= threshold)
							u++;
						else
							v++;
					}
				}

				//System.out.println(p + " " + q + " " + u + " " + v + " " + currentData.size());
				double pi = (p+u)/currentData.size();
				double x = p/(p+u);
				double y = u/(p+u);
				double h = 0;
				if(x!=0)
					h += ((-x)*(log2(x)));
				if(y!=0)
				 	h += ((-y)*(log2(y)));
				//System.out.println(pi + " " + h);
				double entropy = (pi*h);

				pi = (q+v)/currentData.size();
				x = q/(q+v);
				y = v/(q+v);
				h = 0;
				if(x!=0)
					h += ((-x)*(log2(x)));
				if(y!=0)
				 	h += ((-y)*(log2(y)));
				//System.out.println(pi + " " + h);
				entropy += (pi*h);

				if(entropy < bestEntropy) {
					bestEntropy = entropy;
					bestThreshold = threshold;
				}
				//System.out.println("For attr: " + attr + " threshold: " + threshold + " entropy: " + entropy);
			}
			//System.out.println("For attr: " + attr + " bestThreshold: " + bestThreshold + " entropy: " + bestEntropy);
			thresholds.add(bestThreshold);
			entropies.add(bestEntropy);
		}
	}

	private DecTreeNode buildTreeUtil(List<List<Integer>> currentData, int depth) {
		DecTreeNode currentNode = new DecTreeNode(-1, -1, -1);
		int label0 = 0, label1 = 0;
		for(List<Integer> record : currentData) {
			if(getLabel(record) == 0)
				label0++;
			else
				label1++;
		}
		currentNode.classLabel = (label0 > label1) ? 0 : 1;
		if(depth == maxDepth || currentData.size() <= maxPerLeaf || label0 == currentData.size() || label1 == currentData.size())
			return currentNode;

		List<Integer> thresholds = new ArrayList<Integer>();
		List<Double> entropies = new ArrayList<Double>();
		computeEntropies(currentData, entropies, thresholds);

		currentNode.attribute = entropies.indexOf(Collections.min(entropies));
		currentNode.threshold = thresholds.get(currentNode.attribute);
		//System.out.println("BestAttribute: " + currentNode.attribute);
		//System.out.println("bestThreshold: " + currentNode.threshold);

		List<List<Integer>> leftData = new ArrayList<List<Integer>>();
		List<List<Integer>> rightData = new ArrayList<List<Integer>>();
		for(List<Integer> record : currentData) {
			if(record.get(currentNode.attribute) <= currentNode.threshold)
				leftData.add(record);
			else
				rightData.add(record);
		}

		currentNode.left = buildTreeUtil(leftData, depth+1);
		currentNode.right = buildTreeUtil(rightData, depth+1);
		return currentNode;
	}

	private DecTreeNode buildTree() {
		return buildTreeUtil(trainData, 0);
	}

	int classifyUtil(DecTreeNode root, List<Integer> instance) {
		if(root.isLeaf())
			return root.classLabel;
		if(instance.get(root.attribute) <= root.threshold)
			return classifyUtil(root.left, instance);
		return classifyUtil(root.right, instance);
	}

	public int classify(List<Integer> instance) {
		// Note that the last element of the array is the label.
		return classifyUtil(root, instance);
	}

	// Print the decision tree in the specified format
	public void printTree() {
		printTreeNode("", this.root);
	}

	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + "X_" + node.attribute;
		System.out.print(printStr + " <= " + String.format("%d", node.threshold));
		if(node.left.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.left.classLabel));
		}
		else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%d", node.threshold));
		if(node.right.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.right.classLabel));
		}
		else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}
	}

	public double printTest(List<List<Integer>> testDataSet) {
		int numEqual = 0;
		int numTotal = 0;
		for (int i = 0; i < testDataSet.size(); i ++)
		{
			int prediction = classify(testDataSet.get(i));
			int groundTruth = testDataSet.get(i).get(testDataSet.get(i).size() - 1);
			System.out.println(prediction);
			if (groundTruth == prediction) {
				numEqual++;
			}
			numTotal++;
		}
		double accuracy = numEqual*100.0 / (double)numTotal;
		System.out.println(String.format("%.2f", accuracy) + "%");
		return accuracy;
	}
}
